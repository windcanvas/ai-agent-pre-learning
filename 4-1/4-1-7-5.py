#Context Window 边界测试： 写一个脚本，逐步增加输入文本的长度，找到模型开始"忘记前面内容"的临界点。验证是否和官方标注的 Context Window 一致。
"""
Context Window 边界测试（Needle in a Haystack）
思路：
  1) 在输入最开头放一个“唯一暗号 NEEDLE = <随机6位字母+数字>”
  2) 后面跟 N 段填充文本（Haystack），把总长度拉到目标级别
  3) 让模型回答：最开头写的 NEEDLE 是什么
  4) 从短到长逐步增加目标 Token 数，观察从哪个长度开始“说错/答不出”
  5) 对比官方标注的 Context Window 是否一致

前置：
  source "/Users/lql/Downloads/project/ai-agent-pre-learning/.venv/bin/activate"
  export DEEPSEEK_API_KEY="sk-xxxx"
  python3 "/Users/lql/Downloads/project/ai-agent-pre-learning/4-1/4-1-8.py"
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Literal

import requests
import tiktoken

# ========== 基本配置 ==========
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL = "deepseek-chat"
API_URL = "https://api.deepseek.com/chat/completions"

# 为了统计“请求里总 Token 数”，用一套稳定的编码器做近似。
# DeepSeek 官方没公开发布 tokenizer，这里用 gpt-4o(o200k_base) 做“同级别近似”。
# 重点是“同一套编码器跑全流程”，这样趋势是对的。
TOKENIZER_NAME = "o200k_base"
enc = tiktoken.get_encoding(TOKENIZER_NAME)

# 官方标称的上下文窗口（DeepSeek Chat 公开说明是 128k；按你实际模型改）
OFFICIAL_CONTEXT_WINDOW = 128_000

# 测试曲线：从 4k 开始，每次增加 4k，直到接近或超过官方窗口。
# 想更省钱可以把 STEP 调大；想更精细就调小。
START_TOKENS = 4_000
END_TOKENS = 140_000
STEP_TOKENS = 4_000

# 每个长度点跑 2 次，抗随机性（越长越贵，不建议太大）
RUNS_PER_LEVEL = 2

# 输出回复长度限制：让模型只回答 NEEDLE，不要瞎写一大段
MAX_OUTPUT_TOKENS = 128
TEMPERATURE = 0.0  # 温度=0，尽量让它稳定复述，别把“记不住”甩给采样随机性

# 结果保存 JSON
OUTPUT_JSON = Path(__file__).with_name("4-1-8-5-results.json")

# ========== 构建测试输入 ==========
# 一段“有意义但不含暗号”的中文填充语料，越长越好，避免被 tokenizer 特殊处理
HAYSTACK_SENTENCE = (
    "在长上下文能力评测中，研究者通常会在长篇文档的不同位置插入关键信息，"
    "再验证模型是否能在后续任务中准确检索并利用这些信息；常见的插入位置包括文档开头、"
    "文档 25%、50%、75% 以及文档结尾，从而观察不同距离下的召回表现差异。"
)

NEEDLE_RE = re.compile(r"[A-Z0-9]{6}")


def random_needle() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def build_message(needle: str, target_tokens: int) -> str:
    """
    构造单条 user 消息：
      - 最开头：NEEDLE 标记
      - 中间：不断追加填充句，直到整段的 token 数 >= target_tokens
      - 结尾：明确让它回答 NEEDLE
    """
    prefix = (
        "下面是一段长文本。\n"
        "【重要】最开头写了一行暗号，格式如下：\n"
        f"NEEDLE = {needle}\n"
        "请记住这个暗号，后面我会提问。\n\n"
        "—————— 以下是长文本 ——————\n\n"
    )
    suffix = (
        "\n\n—————— 长文本结束 ——————\n\n"
        "请问：长文本最开头写的 NEEDLE 暗号是哪 6 位大写字母/数字？\n"
        "只需要输出这 6 位字符本身，不要输出任何解释、标点或前后缀。"
    )

    # 先算 prefix + suffix 占了多少 token，剩下留给 haystack
    fixed_tokens = count_tokens(prefix) + count_tokens(suffix)
    haystack_target = max(target_tokens - fixed_tokens, 0)

    haystack_parts: list[str] = []
    haystack_tokens = 0
    sentence = HAYSTACK_SENTENCE
    sent_tokens = count_tokens(sentence)

    # 粗追加（快速逼近）
    while haystack_tokens + sent_tokens <= haystack_target:
        haystack_parts.append(sentence)
        haystack_tokens += sent_tokens

    # 细追加：如果还有余量，按句子截断补足（避免过长，也避免离目标差太多）
    # 这里简单处理：差得多就继续加半段；差得少就接受，因为后面会统计“实际 tokens”
    if haystack_tokens < haystack_target:
        haystack_parts.append(sentence)

    haystack = "\n".join(haystack_parts)
    message = prefix + haystack + suffix
    return message


# ========== 调用模型 ==========
def ask_model(user_message: str) -> tuple[str, int | None]:
    """返回 (模型回答内容, 官方返回的 usage.prompt_tokens or None)"""
    if not API_KEY:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严谨的助手。用户要求输出暗号时，只输出暗号本身，不要多说话。"},
            {"role": "user", "content": user_message},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "top_p": 1.0,
        "stream": False,
    }
    r = requests.post(API_URL, headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")
    data = r.json()
    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"解析失败: {e}; raw={r.text[:800]}")
    usage = data.get("usage", {}) or {}
    return answer, usage.get("prompt_tokens")


# ========== 判分 ==========
def extract_answer_candidate(answer: str) -> str | None:
    """从模型回复里抓最像 NEEDLE 的 6 位大写字母数字串"""
    cleaned = answer.strip().upper()
    m = NEEDLE_RE.search(cleaned)
    return m.group(0) if m else None


def score(needle: str, answer: str) -> Literal["correct", "wrong", "refuse"]:
    """
    correct: 明确说对了 6 位暗号
    wrong:   说了一个暗号但不对
    refuse:  答不出/说不知道/完全没提取到候选
    """
    cand = extract_answer_candidate(answer)
    if cand is None:
        return "refuse"
    return "correct" if cand == needle else "wrong"


# ========== 主流程 ==========
def build_levels() -> list[int]:
    levels = list(range(START_TOKENS, END_TOKENS + 1, STEP_TOKENS))
    # 保证官方窗口本身也在曲线上
    if OFFICIAL_CONTEXT_WINDOW not in levels:
        levels.append(OFFICIAL_CONTEXT_WINDOW)
    return sorted(set(levels))


def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "请先设置 DEEPSEEK_API_KEY：\n"
            "  export DEEPSEEK_API_KEY=\"sk-xxxx\""
        )

    levels = build_levels()
    print(f"模型: {MODEL}")
    print(f"官方标称窗口: {OFFICIAL_CONTEXT_WINDOW} tokens")
    print(f"近似编码器: {TOKENIZER_NAME}（用于构造输入长度）")
    print(f"长度点数量: {len(levels)}  每点重复: {RUNS_PER_LEVEL}  预计调用次数: {len(levels) * RUNS_PER_LEVEL}")
    print("=" * 88)

    rows: list[dict] = []

    for target in levels:
        for run in range(1, RUNS_PER_LEVEL + 1):
            needle = random_needle()
            msg = build_message(needle, target)

            # 按我们这套编码器算的“消息总 token 数”
            our_tokens = count_tokens(msg)

            tag = f"[target={target:>6}] run {run}/{RUNS_PER_LEVEL} needle={needle}"
            print(f"{tag}  本地统计≈{our_tokens:>6} tokens ...", end=" ", flush=True)

            try:
                answer, api_prompt_tokens = ask_model(msg)
                result = score(needle, answer)
            except Exception as e:
                # 如果是超长被 API 拒了，这也是“边界”的强信号
                answer = f"[请求失败] {e}"
                result = "error"
                api_prompt_tokens = None

            short_ans = answer.replace("\n", " ")[:80]
            print(f"→ {result:7}  API_prompt_tokens={api_prompt_tokens}  回复: {short_ans}")

            rows.append(
                {
                    "target_tokens": target,
                    "run": run,
                    "needle": needle,
                    "approx_input_tokens": our_tokens,
                    "api_prompt_tokens": api_prompt_tokens,
                    "result": result,
                    "answer": answer,
                }
            )
            time.sleep(0.3)

        # 每过一个长度点就存一次，避免崩了白跑
        OUTPUT_JSON.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ========== 汇总分析 ==========
    print("\n\n" + "=" * 88)
    print("汇总（按目标长度点统计正确率）")
    print("=" * 88)

    header = f"{'target':>8}{'ours≈':>8}{'correct':>9}{'wrong':>7}{'refuse':>8}{'error':>7}{'正确率':>8}"
    print(header)
    print("-" * len(header))

    correctness_curve: list[tuple[int, float]] = []

    for t in levels:
        xs = [r for r in rows if r["target_tokens"] == t]
        if not xs:
            continue
        n = len(xs)
        correct = sum(1 for r in xs if r["result"] == "correct")
        wrong = sum(1 for r in xs if r["result"] == "wrong")
        refuse = sum(1 for r in xs if r["result"] == "refuse")
        error = sum(1 for r in xs if r["result"] == "error")
        our_avg = round(sum(r["approx_input_tokens"] for r in xs) / n)
        acc = correct / n
        correctness_curve.append((t, acc))
        print(f"{t:>8}{our_avg:>8}{correct:>9}{wrong:>7}{refuse:>8}{error:>7}{acc*100:>7.1f}%")

    # 估算“开始遗忘临界点”：连续两个点正确率 < 100% 的第一个点
    first_bad = None
    for i, (t, acc) in enumerate(correctness_curve):
        if acc < 1.0:
            # 允许单点波动：后面至少再跟一个同样 <1.0 的点才算进入遗忘区
            if i + 1 < len(correctness_curve) and correctness_curve[i + 1][1] < 1.0:
                first_bad = t
                break

    # 看看 API 有没有在某个长度直接返回错误（被服务端截断也是一种边界）
    first_error = next((r["target_tokens"] for r in rows if r["result"] == "error"), None)

    print("\n========== 结论 ==========")
    print(f"官方标称 Context Window: {OFFICIAL_CONTEXT_WINDOW} tokens")
    if first_bad is not None:
        print(
            f"实际可稳定复述最开头 NEEDLE 的临界区间: 小于 {first_bad} tokens 基本没问题，"
            f"从约 {first_bad} tokens 开始出现遗忘迹象。"
        )
    else:
        print("在本次测试覆盖的范围内，没有出现系统性遗忘（可考虑把 END_TOKENS 调更大）。")
    if first_error is not None:
        print(
            f"API 在 target≈{first_error} tokens 处返回了错误/拒绝（可能是服务端上下文上限）。"
            "请在 JSON 中查看具体错误信息。"
        )
    print(
        "说明：\n"
        "  1) 这里测的是“开头 Needle”的召回率，属于最容易被挤掉的位置；"
        " 不同位置（开头/25%/50%/75%/结尾）的遗忘曲线可能不同。\n"
        "  2) approx_input_tokens 是用 tiktoken o200k_base 估算的；DeepSeek 官方 tokenizer 与它会有误差，"
        "更可信的是返回的 api_prompt_tokens。曲线对比请以同一口径为准。\n"
        "  3) 完整明细已保存在："
    )
    print(f"     {OUTPUT_JSON}")


if __name__ == "__main__":
    main()