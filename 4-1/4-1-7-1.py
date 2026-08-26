# Token 计数实验： 用 tiktoken 计算以下文本的 Token 数，感受中英文的 Token 效率差异：
# 一段 200 字的中文新闻
# 一段 200 词的英文新闻
# 一段 100 行的 Python 代码

import re
import textwrap
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def compare(text: str, label: str, meta: str = ""):
    tokens = count_tokens(text)
    print(f"========== {label} ==========")
    if meta:
        print(f"说明: {meta}")
    print(f"字符数 (len): {len(text)}")
    print(f"Token 数:     {tokens}")
    print(f"每 Token 字符: {len(text) / tokens:.2f}")
    print()
    return tokens


# 1) 一段约 200 字的中文新闻（实际长度按中文字符数约 200）
cn_news = textwrap.dedent("""
今日，国内人工智能实验室宣布，其自研的大语言模型在新一轮公开评测中表现突出，在中英文理解、数学推理和代码生成等多个任务上刷新榜单成绩。
该团队表示，新模型在训练数据质量、模型架构和对齐方法上均做了系统性优化，同时将推理成本降低约三成。
业内专家认为，随着基础模型能力持续提升，未来在教育、医疗、工业制造等领域的落地应用将进一步加速，但同时也需要更加关注数据安全、算法透明以及模型可解释性等治理议题。
多家企业已表示将基于该模型开发行业智能体产品，预计相关应用将于下半年陆续上线。
""").strip()

# 去掉多余空白后再核对字数（按“字符数”近似衡量中文新闻长度）
cn_news = re.sub(r"\s+", "", cn_news)

# 2) 一段约 200 词的英文新闻（按空格分词，词数接近 200）
en_news = textwrap.dedent("""
A leading domestic artificial intelligence laboratory announced on Tuesday that its newly developed large language model has achieved top results across a new round of public benchmarks, including both Chinese and English natural language understanding, mathematical reasoning, and code generation tasks.
The research team said the model brings systematic improvements in training data quality, model architecture, and alignment methods, while cutting inference costs by roughly thirty percent compared with the previous version.
Industry analysts believe that as the capabilities of foundation models continue to improve, real-world deployment in education, health care, and advanced manufacturing will accelerate significantly in the coming years.
At the same time, they warned that greater attention should be paid to data security, algorithm transparency, model interpretability, and broader AI governance issues.
Several companies already plan to build industry-specific intelligent agents on top of the new model, and the first applications are expected to launch in the second half of this year.
""").strip()

en_words = len(en_news.split())

# 3) 一段 100 行的 Python 代码（刻意覆盖常见语法：函数、类、循环、条件、注释、字符串、列表推导）
py_lines = [
    '"""A small fake Python module for token counting experiments."""',
    "from __future__ import annotations",
    "import math",
    "import json",
    "from dataclasses import dataclass, field",
    "",
    "",
    "DATA_PATH = \"/tmp/data.json\"",
    "MAX_ITER = 1000",
    "EPS = 1e-9",
    "",
    "",
    "@dataclass",
    "class Item:",
    "    name: str",
    "    value: int = 0",
    "    tags: list[str] = field(default_factory=list)",
    "",
    "    def summary(self) -> str:",
    "        return f\"{self.name}:{self.value}\"",
    "",
    "",
    "def load_items(path: str) -> list[Item]:",
    "    # TODO: add logging",
    "    with open(path, \"r\", encoding=\"utf-8\") as f:",
    "        raw = json.load(f)",
    "    items = [Item(**row) for row in raw]",
    "    return items",
    "",
    "",
    "def save_items(items: list[Item], path: str) -> None:",
    "    with open(path, \"w\", encoding=\"utf-8\") as f:",
    "        json.dump([i.__dict__ for i in items], f, ensure_ascii=False, indent=2)",
    "",
    "",
    "def is_prime(n: int) -> bool:",
    "    if n < 2:",
    "        return False",
    "    if n % 2 == 0:",
    "        return n == 2",
    "    limit = int(math.isqrt(n)) + 1",
    "    for d in range(3, limit, 2):",
    "        if n % d == 0:",
    "            return False",
    "    return True",
    "",
    "",
    "def prime_pairs(limit: int) -> list[tuple[int, int]]:",
    "    primes = [x for x in range(2, limit) if is_prime(x)]",
    "    pairs = []",
    "    for i, a in enumerate(primes):",
    "        for b in primes[i + 1:]:",
    "            if b - a == 2:",
    "                pairs.append((a, b))",
    "    return pairs",
    "",
    "",
    "def normalize(values: list[float]) -> list[float]:",
    "    total = sum(values)",
    "    if abs(total) < EPS:",
    "        return [0.0 for _ in values]",
    "    return [v / total for v in values]",
    "",
    "",
    "class Inventory:",
    "    def __init__(self, items: list[Item] | None = None):",
    "        self.items = items or []",
    "",
    "    def add(self, item: Item) -> None:",
    "        self.items.append(item)",
    "",
    "    def total_value(self) -> int:",
    "        return sum(it.value for it in self.items)",
    "",
    "    def filter_by_tag(self, tag: str) -> list[Item]:",
    "        return [it for it in self.items if tag in it.tags]",
    "",
    "    def __len__(self) -> int:",
    "        return len(self.items)",
    "",
    "    def __repr__(self) -> str:",
    "        return f\"Inventory(count={len(self)}, total={self.total_value()})\"",
    "",
    "",
    "def main() -> None:",
    "    items = [",
    "        Item(\"apple\", 10, [\"fruit\", \"fresh\"]),",
    "        Item(\"banana\", 6, [\"fruit\"]),",
    "        Item(\"carrot\", 3, [\"vegetable\"]),",
    "    ]",
    "    inventory = Inventory(items)",
    "    print(inventory)",
    "    print(\"fruits:\", inventory.filter_by_tag(\"fruit\"))",
    "    print(\"twin primes:\", prime_pairs(50))",
    "    print(\"normalized:\", normalize([1.0, 2.0, 3.0, 4.0]))",
    "",
    "",
    "",
    'if __name__ == "__main__":',
    "    main()",
]
assert len(py_lines) == 100, f"py_lines length must be 100, got {len(py_lines)}"
py_code = "\n".join(py_lines)


# 开始统计
print(f"中文新闻字符数（去掉空白）: {len(cn_news)}")
print(f"英文新闻词数: {en_words}")
print(f"Python 代码行数: {len(py_lines)}")
print()

cn_tokens = compare(cn_news, "200 字级别 中文新闻")
en_tokens = compare(en_news, "200 词级别 英文新闻", meta=f"词数约 {en_words}")
py_tokens = compare(py_code, "100 行 Python 代码", meta=f"行数 {len(py_lines)}")

# 直观对比
print("========== 对比总结 ==========")
print(f"1 个 Token ≈ {len(cn_news) / cn_tokens:.2f} 个中文字符")
print(f"1 个 Token ≈ {en_words / en_tokens:.2f} 个英文词")
print(f"1 个 Token ≈ {len(py_lines) / py_tokens:.2f} 行 Python 代码")
print(f"相同信息量下，中文 Token 数 / 英文 Token 数 ≈ {cn_tokens / en_tokens:.2f}x")