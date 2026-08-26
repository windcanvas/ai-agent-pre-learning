import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

def check_context_fit(system_prompt: str, messages: list[str], max_tokens: int = 128000):
    """检查所有内容是否在上下文窗口内"""
    total = len(enc.encode(system_prompt))
    for msg in messages:
        total += len(enc.encode(msg))

    usage_pct = total / max_tokens * 100

    print(f"System Prompt: {len(enc.encode(system_prompt))} tokens")
    print(f"对话历史: {len(enc.encode(''.join(messages)))} tokens")
    print(f"总计: {total} / {max_tokens} tokens ({usage_pct:.1f}%)")

    if total > max_tokens:
        print(f"⚠️ 超出上限！请缩短内容或使用上下文压缩")
    else:
        print(f"✅ 安全范围内")

# 模拟一次 Agent 对话
system_prompt = "你是一个专业的代码审查助手..." * 100  # 一个比较长的系统提示
messages = [
    "帮我审查这段代码：def foo(): pass",
    "这段代码有问题吗？",
    "能否给出具体的修改建议？",
]
check_context_fit(system_prompt, messages)
