"""
Temperature 效果观察：同一个 Prompt 分别用 0 / 0.5 / 1.0 / 1.5 调用 DeepSeek API
用法：
  1) 把 DEEPSEEK_API_KEY 换成你自己的 Key，或者先设置环境变量：
       export DEEPSEEK_API_KEY="sk-xxxx"
  2) 运行：
       source "/Users/lql/Downloads/project/ai-agent-pre-learning/.venv/bin/activate"
       python3 "/Users/lql/Downloads/project/ai-agent-pre-learning/4-1/4-1-6.py"
"""

import json
import os
import time
import requests

# ============ 基本配置 ============
API_KEY = os.getenv("DEEPSEEK_API_KEY", "") # 也可以直接写在这里
MODEL = "deepseek-chat"
API_URL = "https://api.deepseek.com/chat/completions"

# 同一个 Prompt，故意选"既需要知识、又有发挥空间"的题
PROMPT = (
    "请用中文写一段 150 字左右的科普短文，题目是："
    "《为什么夏天会比冬天更容易看到闪电？》。"
    "要求：语言活泼、适合中学生阅读，结尾给一个有趣的小思考题。"
)

TEMPERATURES = [0.0, 0.5, 1.0, 1.5]
MAX_TOKENS = 400
TOP_P = 1.0
N_PER_TEMP = 3  # 每个温度跑 3 次，便于观察"稳定性"


def call_deepseek(prompt: str, temperature: float, timeout: int = 60) -> dict:
    """调一次 DeepSeek Chat Completions，返回原始 JSON"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一位擅长讲解科学知识的中文老师。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "stream": False,
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"DeepSeek API 返回 {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


def extract_answer(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        return f"[解析失败: {e}] 原始响应: {json.dumps(data, ensure_ascii=False)[:200]}"


def similarity(a: str, b: str) -> float:
    """
    极简"稳定性"近似：比较两个答案的字符级别 Jaccard 相似度。
    相似度越高 ⇒ 两次输出越像 ⇒ 那个温度更稳定。
    生产上会用更专业的（比如 embedding 余弦相似度、ROUGE），这里做演示够用。
    """
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "请先设置 DEEPSEEK_API_KEY，比如：\n"
            "  export DEEPSEEK_API_KEY=\"sk-xxxx\"\n"
            "或者直接改脚本里的 API_KEY 常量。"
        )

    results = {}  # temperature -> list[str]

    for temp in TEMPERATURES:
        answers = []
        print(f"\n===== Temperature = {temp} =====")
        for i in range(N_PER_TEMP):
            print(f"  第 {i + 1}/{N_PER_TEMP} 次调用...", end=" ", flush=True)
            try:
                data = call_deepseek(PROMPT, temperature=temp)
                ans = extract_answer(data)
                answers.append(ans)
                print(f"成功（长度 {len(ans)} 字符）")
            except Exception as e:
                print(f"失败: {e}")
                answers.append(f"[调用失败: {e}]")
            # 别打太快，避免触发速率限制
            time.sleep(0.5)
        results[temp] = answers

    # ====== 对比展示 ======
    print("\n\n" + "=" * 60)
    print("输出对比（每个温度跑 3 次）")
    print("=" * 60)
    for temp, answers in results.items():
        print(f"\n------------- Temperature = {temp} -------------")
        for idx, ans in enumerate(answers, 1):
            print(f"\n— 第 {idx} 次 —")
            print(ans)

    # ====== 稳定性评估 ======
    print("\n\n" + "=" * 60)
    print("稳定性评估：3 次答案之间的两两相似度（越高越稳定）")
    print("=" * 60)

    rows = []
    for temp, answers in results.items():
        valid = [a for a in answers if not a.startswith("[调用失败")]
        if len(valid) < 2:
            avg_sim = None
            note = "(可用样本不足，跳过)"
        else:
            sims = []
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    sims.append(similarity(valid[i], valid[j]))
            avg_sim = sum(sims) / len(sims)
            note = ""
        rows.append((temp, avg_sim, note))
        print(
            f"Temperature = {temp:<4}  平均相似度 = "
            + (f"{avg_sim * 100:.1f}%  {note}" if avg_sim is not None else f"N/A  {note}")
        )

    # ====== 结论 ======
    valid_rows = [(t, s, n) for t, s, n in rows if s is not None]
    if valid_rows:
        most_stable = max(valid_rows, key=lambda x: x[1])
        most_creative = min(valid_rows, key=lambda x: x[1])
        print("\n\n" + "=" * 60)
        print("结论（基于本次 3 次调用）")
        print("=" * 60)
        print(f"最稳定：Temperature = {most_stable[0]}，平均相似度 {most_stable[1] * 100:.1f}%")
        print(f"最有创意（最不稳定）：Temperature = {most_creative[0]}，平均相似度 {most_creative[1] * 100:.1f}%")
        print(
            "\n直觉上的解释：\n"
            "  - Temperature = 0 几乎每次都选最高分词，所以 3 次内容几乎一模一样，最稳定，但也最容易无聊。\n"
            "  - Temperature = 1.5 给冷门词更多概率，3 次用词、结构、例子差异都最大，创意最强，但也最容易跑题/啰嗦。\n"
            "  - 0.5 / 1.0 通常是兼顾准确性和可读性的折中。"
        )


if __name__ == "__main__":
    main()