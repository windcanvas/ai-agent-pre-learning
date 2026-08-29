# 一次 Agent 推理的完整 Context —— messages 数组的每一段
def build_context(task, history, tools, memory, env):
    """构建发送给 LLM 的完整上下文"""
    messages = []

    # 第 1 层: System Prompt（行为规则 + 工具定义）
    messages.append({
        "role": "system",
        "content": f"""你是 AI Agent。

## 工具
{format_tools(tools)}

## 规则
- 需要工具时输出: <tool> 名称 </tool><params>JSON 参数 </params>
- 完成任务后输出: <done> 结果 </done>
- 不确定时主动向用户确认，不要瞎猜
- 工具调用失败时尝试替代方案，最多重试 2 次""",
    })

    # 第 2 层: 任务状态（当前进度 + 环境信息）
    messages.append({
        "role": "system",
        "content": f"""[任务状态]
当前时间: {env['now']}
工作目录: {env['cwd']}
已执行步骤: {len(history)} 步
API 额度: 剩余 {env['api_quota']}""",
    })

    # 第 3 层: 用户记忆（偏好 + 历史）
    if memory:
        messages.append({
            "role": "system",
            "content": f"[用户偏好]\n{format_memory(memory)}",
        })

    # 第 4 层: 对话历史（最近的工具调用和结果）
    messages.extend(history)

    # 第 5 层: 当前任务
    messages.append({"role": "user", "content": task})

    return messages
