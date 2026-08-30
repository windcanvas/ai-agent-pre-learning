# Context 构建： 一个 Agent 要执行"分析 GitHub 仓库 pytorch 最近一周的 Issue 趋势"。写出它的 Context（messages 数组）应包含哪些内容，每类内容占多少 Token。
"""
Context 构建示例：分析 GitHub 仓库 pytorch 最近一周的 Issue 趋势
按照五层上下文组装模式：
Layer 1: System 规则 + 工具定义  (~3,200 tokens, 首因效应区)
Layer 2: 任务状态/环境元数据      (~120 tokens)
Layer 3: 用户偏好/长期记忆        (~80 tokens)
Layer 4: 对话历史                 (~0 tokens, 首轮对话)
Layer 5: 当前任务                 (~60 tokens)

合计预估: ~3,460 tokens
Token 估算方式: 中文字符 ~1.5 chars/token, 英文 ~4 chars/token, 粗略取 len//2
"""

import json
from datetime import datetime, timedelta
import os


def build_github_analysis_context(user_task: str) -> tuple[list[dict], dict]:
    """
    构建执行 GitHub Issue 趋势分析任务的 Context
    
    Returns:
        (messages 数组, 每层 Token 统计)
    """

    # ==================== Layer 1: System 规则 + 工具定义 (~3,200 tokens) ====================
    # 放在最顶层，利用首因效应强化模型对规则和工具的服从度
    system_rule_and_tools = """## 身份与目标
你是一个专业的开源社区数据分析 Agent，擅长从 GitHub 数据中提取有价值的趋势洞察。

## 核心行为规则
1. 所有数据获取必须通过工具调用，禁止编造数据或凭空推测
2. 调用工具前先检查参数是否完整，特别是时间范围和仓库路径
3. 分页数据必须完整拉取，不能只取第一页就下结论
4. 分析结论必须有数据支撑，标注数据来源时间窗口
5. 如果工具调用连续失败 2 次，停止重试并向用户说明障碍

## 输出格式协议
### 工具调用格式（使用 <tool> 自定义标签）
<tool>
{
  "name": "工具名称",
  "params": {"参数名": "参数值"}
}
</tool>

### 任务完成格式（使用 <done> 自定义标签）
<done>
{
  "summary": "分析结论摘要",
  "key_findings": ["发现1", "发现2"],
  "data_coverage": "数据覆盖范围说明"
}
</done>

## 可用工具定义
<tools>
[
  {
    "name": "github_list_issues",
    "description": "列出指定仓库在给定时间范围内的 Issues，支持分页",
    "params": {
      "owner": {"type": "string", "required": true, "description": "仓库所有者，如 pytorch"},
      "repo": {"type": "string", "required": true, "description": "仓库名称，如 pytorch"},
      "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "all"},
      "since": {"type": "string", "required": true, "description": "ISO 格式开始时间，如 2026-08-23T00:00:00Z"},
      "until": {"type": "string", "description": "ISO 格式结束时间，默认当前时间"},
      "per_page": {"type": "int", "default": 100, "max": 100},
      "page": {"type": "int", "default": 1, "description": "页码，从 1 开始"}
    },
    "returns": {
      "total_count": "总 Issue 数",
      "page_info": {"has_next_page": "bool", "next_cursor": "string"},
      "items": [
        {
          "number": "Issue 编号",
          "title": "标题",
          "state": "open/closed",
          "labels": ["bug", "enhancement", "module: cuda", ...],
          "created_at": "创建时间 ISO",
          "updated_at": "更新时间 ISO",
          "closed_at": "关闭时间 ISO 或 null",
          "comments": "评论数",
          "user": {"login": "创建者用户名", "type": "User/Bot"}
        }
      ]
    }
  },
  {
    "name": "github_get_issue_comments",
    "description": "获取单个 Issue 的评论内容，用于高频关键词抽取",
    "params": {
      "owner": {"type": "string", "required": true},
      "repo": {"type": "string", "required": true},
      "issue_number": {"type": "int", "required": true},
      "per_page": {"type": "int", "default": 50}
    },
    "returns": {
      "comments": [
        {"body": "评论正文", "created_at": "时间", "user": "用户名"}
      ]
    }
  },
  {
    "name": "data_aggregate_issues",
    "description": "对 Issue 列表做聚合统计：按天分布、标签分布、状态分布、Top 关键词",
    "params": {
      "issues": {"type": "array", "required": true, "description": "github_list_issues 返回的 items"},
      "timezone": {"type": "string", "default": "UTC"}
    },
    "returns": {
      "daily_created": {"2026-08-23": 15, "2026-08-24": 22, ...},
      "daily_closed": {"2026-08-23": 8, ...},
      "label_distribution": {"bug": 45, "enhancement": 23, "module: cuda": 18, ...},
      "top_creator_users": [{"login": "xxx", "count": 5}],
      "high_priority_count": "含 high priority 或 critical 标签的 Issue 数",
      "avg_hours_to_close": "已关闭 Issue 平均处理时长(小时)"
    }
  },
  {
    "name": "trend_compare",
    "description": "对比本周与上周的趋势变化，计算增长率",
    "params": {
      "current_period": {"created": 120, "closed": 80, "bug_count": 45},
      "previous_period": {"created": 105, "closed": 88, "bug_count": 38}
    },
    "returns": {
      "created_growth_rate": "+14.3%",
      "closed_growth_rate": "-9.1%",
      "bug_trend": "上升 18.4%",
      "backlog_change": "积压 +28 issues"
    }
  }
]
</tools>

## 标准分析框架（参考）
1. 数据拉取：确认时间窗口 → 拉取全部 Issue（注意分页）
2. 基础指标：创建量、关闭量、积压变化、平均处理时长
3. 维度下钻：标签分布、模块分布、Top 创作者、Bot vs 真人比例
4. 趋势对比：同比上周 / 过去四周移动平均
5. 异常检测：单日尖峰、高频关键词、关键模块异动
"""

    # ==================== Layer 2: 任务状态/环境元数据 (~120 tokens) ====================
    now = datetime.utcnow()
    one_week_ago = now - timedelta(days=7)
    env_metadata = f"""## 任务环境元信息
<env>
K: 执行时间(now) | V: {now.isoformat()}Z
K: 分析窗口起始(since) | V: {one_week_ago.strftime('%Y-%m-%dT00:00:00Z')}
K: 分析窗口结束(until) | V: {now.isoformat()}Z
K: 目标仓库(owner/repo) | V: pytorch/pytorch
K: 时区(timezone) | V: UTC
K: 工作目录(cwd) | V: {os.getcwd()}
K: GitHub API 状态 | V: 正常，可用额度充足
K: 执行模式 | V: 实时拉取 + 本地聚合
</env>"""

    # ==================== Layer 3: 用户偏好/长期记忆 (~80 tokens) ====================
    user_memory = """## 用户偏好记忆
<memory>
K: 分析粒度偏好 | V: 同时提供宏观趋势 + Top 5 异常点
K: 标签关注优先级 | V: module: cuda > triage review > bug > high priority
K: 输出语言偏好 | V: 中文分析 + 英文标签原文
K: 数据可信度要求 | V: 样本量<30 时标注为「参考值」
K: 过往关注点 | V: CUDA 相关 Issue 激增、回归类 Bug、首响时长
</memory>"""

    # ==================== Layer 4: 对话历史 (~0 tokens, 首轮对话) ====================
    # 首次执行任务，无历史对话需要注入

    # ==================== Layer 5: 当前用户任务 (~60 tokens) ====================
    current_task = f"""## 当前任务
<task>
目标：分析 GitHub 仓库 pytorch 最近一周的 Issue 趋势
时间范围：{one_week_ago.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')}（UTC，最近 7 天）
要求：
1. 给出创建/关闭趋势及积压变化
2. 识别 Top 3 增长最快的标签和模块
3. 标注是否有异常尖峰及可能原因
4. 与上周对比给出核心结论
</task>"""

    # ==================== 组装 messages 数组 ====================
    messages = [
        {"role": "system", "content": system_rule_and_tools},   # L1: 规则+工具（顶层→首因效应）
        {"role": "system", "content": env_metadata},            # L2: 环境元数据
        {"role": "system", "content": user_memory},             # L3: 用户记忆
        # L4: 对话历史（空，在 run 循环中动态追加 assistant/tool 消息）
        {"role": "user", "content": current_task},              # L5: 当前任务（近因效应区）
    ]

    # ==================== Token 统计（粗略估算：len // 2） ====================
    token_stats = {
        "layer_1_system_rules_and_tools": {
            "chars": len(system_rule_and_tools),
            "est_tokens": len(system_rule_and_tools) // 2,
            "description": "System 规则 + 4 个工具定义"
        },
        "layer_2_env_metadata": {
            "chars": len(env_metadata),
            "est_tokens": len(env_metadata) // 2,
            "description": "实时环境元信息（时间、窗口、仓库、额度）"
        },
        "layer_3_user_memory": {
            "chars": len(user_memory),
            "est_tokens": len(user_memory) // 2,
            "description": "用户偏好与历史关注点"
        },
        "layer_4_chat_history": {
            "chars": 0,
            "est_tokens": 0,
            "description": "对话历史（首轮为空，执行中增长）"
        },
        "layer_5_current_task": {
            "chars": len(current_task),
            "est_tokens": len(current_task) // 2,
            "description": "当前用户任务指令"
        },
    }
    total_chars = sum(v["chars"] for v in token_stats.values())
    total_tokens = sum(v["est_tokens"] for v in token_stats.values())
    token_stats["_total"] = {
        "chars": total_chars,
        "est_tokens": total_tokens,
        "window_usage": f"{total_tokens}/128000 = {total_tokens/1280:.2f}%"
    }

    return messages, token_stats


if __name__ == "__main__":
    task = "分析 GitHub 仓库 pytorch 最近一周的 Issue 趋势"
    messages, stats = build_github_analysis_context(task)

    print("=" * 60)
    print("Context (messages) 结构总览")
    print("=" * 60)
    for i, msg in enumerate(messages):
        preview = msg["content"].split("\n")[0][:80]
        print(f"[{i}] role={msg['role']:<8}  首行预览: {preview}...")

    print("\n" + "=" * 60)
    print("Token 预估统计 (粗略估算: chars // 2)")
    print("=" * 60)
    for k, v in stats.items():
        if k.startswith("_"):
            print(f"\n>>> {k.strip('_').upper()} <<<")
            print(f"  字符总数:    {v['chars']:,} chars")
            print(f"  Token 预估:  {v['est_tokens']:,} tokens")
            print(f"  128k 窗口占用: {v['window_usage']}")
        else:
            print(f"\n{k}")
            print(f"  描述:    {v['description']}")
            print(f"  字符数:  {v['chars']:>6,} chars")
            print(f"  Token:   {v['est_tokens']:>5,} tokens")

    print("\n" + "=" * 60)
    print("完整 messages JSON (已导出到 context_output.json)")
    print("=" * 60)
    with open(
        "/Users/lql/Downloads/project/ai-agent-pre-learning/4-10/4_10_7_1_context_output.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump({"messages": messages, "token_stats": stats}, f,
                  ensure_ascii=False, indent=2)
    print("✅ 已写入: /Users/lql/Downloads/project/ai-agent-pre-learning/4-10/context_output.json")