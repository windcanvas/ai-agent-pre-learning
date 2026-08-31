#工具设计： 为"自动代码审查 Agent"设计 4 个工具，每个写清楚名称、描述、参数（JSON Schema 格式）、安全级别。
"""
工具设计：自动代码审查 Agent
4 个工具形成完整闭环：拉取变更 → 静态扫描 → 质量度量 → 提交评论
每个工具包含: name / description / params (JSON Schema) / security_level
安全级别定义:
  SAFE    - 纯只读/纯计算，无副作用，可无条件调用
  CAUTION - 读取敏感信息或依赖外部资源/有配额，需鉴权/限流
  UNSAFE  - 有写入/修改副作用，调用前必须有人工确认或触发审批流程
"""

from enum import Enum


class SecurityLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    UNSAFE = "UNSAFE"


# ==============================================================================
# 工具注册表（4 个工具，覆盖审查全链路）
# ==============================================================================
CODE_REVIEW_TOOLS = [

    # --------------------------------------------------------------------------
    # Tool 1 / 4: git_fetch_pr_diff
    # 用途：审查入口工具 —— 先知道"改了什么"才有分析对象
    # --------------------------------------------------------------------------
    {
        "name": "git_fetch_pr_diff",
        "description": (
            "从代码托管平台（GitHub/GitLab/Gitea）获取指定 Pull Request / Merge Request "
            "的完整文件变更差异（Diff）。返回每个变更文件的路径、增删行统计、以及统一格式的 "
            "diff hunk 内容。审查 Agent 的第一个动作必须调用此工具拿到原始变更集。"
        ),
        "security_level": SecurityLevel.CAUTION,
        "security_reason": (
            "需要仓库 READ 权限；大 PR 的 Diff 可能消耗大量 API 额度（单 PR 上限 ~5000 文件）。"
            "不会修改任何数据，但需注意 API Rate Limit。"
        ),
        "params": {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "required": ["platform", "repo_owner", "repo_name", "change_id"],
            "additionalProperties": False,
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["github", "gitlab", "gitea", "bitbucket"],
                    "description": "代码托管平台，决定 API 路由格式",
                },
                "repo_owner": {
                    "type": "string",
                    "pattern": r"^[a-zA-Z0-9_.-]+$",
                    "description": "仓库所有者（组织/用户名），如 pytorch",
                },
                "repo_name": {
                    "type": "string",
                    "pattern": r"^[a-zA-Z0-9_.-]+$",
                    "description": "仓库名称，如 pytorch",
                },
                "change_id": {
                    "oneOf": [
                        {"type": "integer", "minimum": 1},
                        {"type": "string", "pattern": r"^[a-fA-F0-9]{7,40}$"},
                    ],
                    "description": "PR/MR 编号（int）或 Commit SHA（7~40 位十六进制）",
                },
                "file_path_include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "只拉取匹配 glob 模式的文件路径，如 ['src/**/*.py', '!tests/**']",
                },
                "max_files": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 200,
                    "description": "单次最大拉取文件数，超限返回部分 diff 并在 meta 中标记 truncated",
                },
                "max_lines_per_file": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 5000,
                    "default": 1000,
                    "description": "单文件最大行数，超限截断并在该文件上标记 truncated",
                },
                "include_context_lines": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 50,
                    "default": 3,
                    "description": "每个 diff hunk 前后保留的上下文行数，帮助理解代码结构",
                },
            },
        },
        "returns_example": {
            "meta": {
                "change_id": 12345,
                "title": "优化 CUDA kernel 启动逻辑",
                "author": "gh-user-01",
                "total_files": 42,
                "returned_files": 42,
                "truncated": False,
            },
            "files": [
                {
                    "path": "torch/cuda/device.py",
                    "status": "modified",  # added / modified / removed / renamed
                    "additions": 56,
                    "deletions": 21,
                    "diff_hunks": [
                        {
                            "old_start": 100, "old_lines": 15,
                            "new_start": 100, "new_lines": 28,
                            "unified_diff": "@@ -100,15 +100,28 @@ ...",
                        },
                    ],
                },
            ],
        },
    },

    # --------------------------------------------------------------------------
    # Tool 2 / 4: static_analyze_snippet
    # 用途：问题发现工具 —— 在 LLM 人工审查前先扫出确定性问题
    # --------------------------------------------------------------------------
    {
        "name": "static_analyze_snippet",
        "description": (
            "对传入的代码片段执行多维度静态分析：语法合法性检查、代码风格匹配、潜在 Bug 模式识别、"
            "安全漏洞规则匹配（OWASP Top 10 / CWE 常见模式）。纯本地规则引擎，无网络调用。"
            "返回每个发现的问题及其具体行号、严重级别、修复建议片段。"
        ),
        "security_level": SecurityLevel.SAFE,
        "security_reason": (
            "纯本地计算的规则匹配，无文件写入、无网络调用、不执行任何传入代码。"
            "潜在风险：大段代码消耗 CPU（通过 maxLength 参数限制）；custom_rules 中的正则"
            "可能引起灾难性回溯(ReDoS)，运行时需设置匹配超时(如 2s)并限制 pattern 长度。"
        ),
        "params": {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "required": ["code_snippet", "language"],
            "additionalProperties": False,
            "properties": {
                "code_snippet": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200000,  # 约 20 万字符上限
                    "description": "待分析的源代码全文或片段（建议单次 <= 5000 行）",
                },
                "language": {
                    "type": "string",
                    "enum": [
                        "python", "python3", "javascript", "typescript",
                        "go", "rust", "java", "kotlin", "cpp", "c",
                        "csharp", "ruby", "php", "shell", "sql",
                    ],
                    "description": "编程语言，决定加载哪套规则集和语法解析器",
                },
                "file_path": {
                    "type": "string",
                    "description": "代码所属文件路径，用于启发式规则（如 __init__.py 特殊处理）",
                },
                "rule_sets": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "syntax",       # 语法错误
                            "style",        # PEP8 / ESLint / gofmt 等风格
                            "bug_pattern",  # ==None、可变默认参数、资源泄漏 等
                            "security",     # 注入、硬编码密钥、危险反序列化 等
                            "performance",  # N+1 查询、大循环内 IO 等
                            "type_check",   # 基本类型推断（非 mypy 级严格）
                        ],
                    },
                    "default": ["syntax", "bug_pattern", "security"],
                    "description": "启用的规则集合，默认只开确定性高的三项以降低误报",
                },
                "severity_threshold": {
                    "type": "string",
                    "enum": ["info", "warning", "error", "critical"],
                    "default": "warning",
                    "description": "只返回 >= 此级别的问题，过滤掉 info 级噪音",
                },
                "custom_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "pattern", "severity", "message"],
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "pattern": {
                                "type": "string",
                                "maxLength": 500,
                                "description": "正则表达式，匹配代码行（引擎需开启超时保护以防 ReDoS）",
                            },
                            "severity": {"type": "string", "enum": ["info", "warning", "error", "critical"]},
                            "message": {"type": "string", "description": "命中时的报错文案"},
                        },
                    },
                    "default": [],
                    "description": "项目私有规则（如禁止 torch.cuda.synchronize 热路径调用）",
                },
            },
        },
        "returns_example": {
            "summary": {"total": 5, "critical": 1, "error": 2, "warning": 2},
            "findings": [
                {
                    "rule_id": "SEC-HARDCODED-KEY",
                    "line": 42,
                    "column": 15,
                    "severity": "critical",
                    "rule_set": "security",
                    "message": "疑似硬编码 AWS Access Key，应移至环境变量",
                    "snippet": "AWS_KEY = \"AKIAIOSFODNN7EXAMPLE\"",
                    "fix_suggestion": "AWS_KEY = os.environ[\"AWS_ACCESS_KEY_ID\"]",
                },
            ],
        },
    },

    # --------------------------------------------------------------------------
    # Tool 3 / 4: measure_code_quality
    # 用途：量化度量工具 —— 提供可比较的数字指标，而非仅主观文字评论
    # --------------------------------------------------------------------------
    {
        "name": "measure_code_quality",
        "description": (
            "对变更文件集合计算代码质量的量化维度：圈复杂度(Cyclomatic Complexity)、"
            "认知复杂度(Cognitive Complexity)、函数长度分布、重复代码率、"
            "入站/出站依赖变化、测试覆盖率缺口（如果仓库有 baseline 文件）。"
            "返回结构化度量结果和与仓库基线对比的「异常红灯」列表。"
        ),
        "security_level": SecurityLevel.CAUTION,
        "security_reason": (
            "只读操作，但在复杂项目中可能触发 radon / clippy / eslint 等本地命令执行；"
            "需在沙箱环境运行，禁止允许通过此工具执行任意用户可控命令。"
            "另外会读取项目中的 .quality_baseline.json 基线文件（如果存在）。"
        ),
        "params": {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "required": ["file_metrics_requests"],
            "additionalProperties": False,
            "properties": {
                "file_metrics_requests": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 200,
                    "items": {
                        "type": "object",
                        "required": ["path", "language", "code_content"],
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string", "description": "文件相对路径"},
                            "language": {
                                "type": "string",
                                "enum": [
                                    "python", "python3", "javascript", "typescript",
                                    "go", "rust", "java", "kotlin", "cpp", "c",
                                    "csharp", "ruby", "php", "shell", "sql",
                                ],
                            },
                            "code_content": {"type": "string"},
                        },
                    },
                    "description": "待度量的文件列表数组，每项包含路径+语言+全文",
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "cyclomatic_complexity",   # 圈复杂度
                            "cognitive_complexity",    # 认知复杂度
                            "function_length_stats",   # 函数长度分布
                            "duplication_rate",        # 重复代码率
                            "dependency_change",       # 依赖变化
                            "coverage_gap",            # 覆盖率缺口（需 baseline）
                            "comment_ratio",           # 注释率
                            "naming_consistency",      # 命名一致性
                        ],
                    },
                    "default": [
                        "cyclomatic_complexity",
                        "function_length_stats",
                        "duplication_rate",
                        "coverage_gap",
                    ],
                },
                "repo_baseline_path": {
                    "type": "string",
                    "description": "仓库历史基线 JSON 文件的本地路径（包含 avg_complexity 等），如 .quality_baseline.json。安全约束：仅允许仓库内相对路径，实现层需做白名单/目录越界校验，禁止读取任意本地文件",
                },
                "threshold_overrides": {
                    "type": "object",
                    "description": "覆盖默认阈值，如 {\"max_function_complexity\": 15, \"max_duplication_pct\": 5}",
                    "default": {},
                    "additionalProperties": False,
                    "properties": {
                        "max_function_complexity": {"type": "integer", "minimum": 1},
                        "max_function_lines": {"type": "integer", "minimum": 1},
                        "max_duplication_pct": {"type": "number", "minimum": 0, "maximum": 100},
                        "min_tests_per_new_file": {"type": "integer", "minimum": 0},
                    },
                },
            },
        },
        "returns_example": {
            "files_analyzed": 12,
            "baseline_compared": True,
            "per_file": [
                {
                    "path": "torch/cuda/device.py",
                    "avg_cyclomatic": 7.2,
                    "max_cyclomatic": 21,  # 红灯
                    "max_cyclomatic_function": "_launch_kernel_grid",
                    "duplication_pct": 3.1,
                    "hotspots": ["function _launch_kernel_grid complexity=21 exceeds limit 12"],
                },
            ],
            "summary_red_lights": [
                {
                    "severity": "warning",
                    "message": "1 个函数圈复杂度(21)超过基线(12)，建议拆分",
                },
            ],
        },
    },

    # --------------------------------------------------------------------------
    # Tool 4 / 4: submit_review_comments
    # 用途：结果出口工具 —— 把审查结论写回代码平台，有写入副作用
    # --------------------------------------------------------------------------
    {
        "name": "submit_review_comments",
        "description": (
            "将审查结果批量写回代码托管平台：逐行评论（Inline Comment）、"
            "PR 整体总结评论（PR Body Comment）、以及审查结论（Approve / Request Changes / Comment）。"
            "写入前会先执行幂等性检查：同一 Agent ID + 同一 PR 编号默认只保留最近一次评论集。"
            "此工具具有写入副作用，调用前必须通过人类确认前置或审批 Gate。"
        ),
        "security_level": SecurityLevel.UNSAFE,
        "security_reason": (
            "会真实写入评论到平台，影响团队协作流。错误的评论或过高噪音会骚扰同事、"
            "污染 PR 讨论上下文。特别是 Approve 操作会直接影响合并权限，必须严格审批。"
            "建议生产环境配置：REQUIRE_HUMAN_APPROVAL_BEFORE_SUBMIT=true。"
        ),
        "params": {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "required": [
                "platform", "repo_owner", "repo_name", "change_id",
                "reviewer_agent_id", "review_action", "comments",
            ],
            "additionalProperties": False,
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["github", "gitlab", "gitea", "bitbucket"],
                },
                "repo_owner": {
                    "type": "string",
                    "pattern": r"^[a-zA-Z0-9_.-]+$",
                },
                "repo_name": {
                    "type": "string",
                    "pattern": r"^[a-zA-Z0-9_.-]+$",
                },
                "change_id": {
                    "oneOf": [
                        {"type": "integer", "minimum": 1},
                        {"type": "string", "pattern": r"^[a-fA-F0-9]{7,40}$"},
                    ],
                },
                "reviewer_agent_id": {
                    "type": "string",
                    "pattern": r"^[a-zA-Z0-9_-]{3,64}$",
                    "description": "Agent 唯一标识，用于评论署名和幂等去重，如 cr-bot-pytorch-v3",
                },
                "review_action": {
                    "type": "string",
                    "enum": ["APPROVE", "REQUEST_CHANGES", "COMMENT"],
                    "description": (
                        "审查结论动作。REQUEST_CHANGES 会阻止合并直至问题解决。"
                        "APPROVE 只允许在 0 critical + 0 error 发现时使用（工具内部强制校验）。"
                    ),
                },
                "comments": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 200,
                    "items": {
                        "type": "object",
                        "required": ["body"],
                        "additionalProperties": False,
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "文件相对路径。为空 = PR 整体评论而非逐行评论",
                            },
                            "line": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "评论锚定的新文件行号（side=RIGHT）。path 存在时必填",
                            },
                            "side": {
                                "type": "string",
                                "enum": ["LEFT", "RIGHT"],
                                "default": "RIGHT",
                                "description": "挂在旧版本还是新版本行号上",
                            },
                            "body": {
                                "type": "string",
                                "minLength": 5,
                                "maxLength": 5000,
                                "description": "评论正文（Markdown 格式）。以 AI-Generated 标记开头以透明披露",
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["nit", "suggestion", "warning", "error", "critical"],
                                "default": "suggestion",
                            },
                            "auto_suggest_patch": {
                                "type": "string",
                                "description": (
                                    "可选：GitHub Suggestion Patch 格式的修复代码。"
                                    "填写后评论中会出现「接受建议」一键提交按钮，风险更高。"
                                ),
                            },
                            "sources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "证据来源，如 ['static_analyze:SEC-HARDCODED-KEY', 'llm_review:R-7']",
                            },
                        },
                    },
                },
                "summary_body": {
                    "type": "string",
                    "maxLength": 20000,
                    "description": "PR 级总结评论正文（放在最上方，含概览表格 + 红灯清单）",
                },
                "idempotency": {
                    "type": "object",
                    "default": {"overwrite_previous_by_agent": True, "dedup_window_hours": 24},
                    "additionalProperties": False,
                    "properties": {
                        "overwrite_previous_by_agent": {
                            "type": "boolean",
                            "description": "是否先删除同一 reviewer_agent_id 的上一轮评论再提交新的",
                        },
                        "dedup_window_hours": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "幂等窗口：同一 PR 在此时间内重复调用只生效一次",
                        },
                    },
                },
                "human_approval_token": {
                    "type": "string",
                    "minLength": 16,
                    "description": (
                        "人工审批通过的一次性 Token。dev/test 环境可省略；生产环境必填，"
                        "缺失或无效 Token 时工具直接返回 403 拒绝写入。"
                        "是否强制由服务端配置 REQUIRE_HUMAN_APPROVAL_BEFORE_SUBMIT 决定。"
                    ),
                },
            },
        },
        "returns_example": {
            "status": "submitted",
            "review_action": "REQUEST_CHANGES",
            "submitted_comments": 23,
            "deleted_previous_comments": 19,
            "links": {
                "pr_url": "https://github.com/pytorch/pytorch/pull/12345",
                "review_id": 2184930712,
            },
        },
    },

]


# ==============================================================================
# 格式转换：导出为 OpenAI Function Calling 兼容结构
# ==============================================================================
def to_openai_function_specs():
    """把工具清单转换为 OpenAI Function Calling 格式。

    标准格式只接受 name / description / parameters 三个字段，
    本文件额外携带的 security_level / security_reason / returns_example
    会在转换时丢弃（OpenAI SDK 不接受未知字段）。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["params"],
            },
        }
        for tool in CODE_REVIEW_TOOLS
    ]


# ==============================================================================
# 打印工具清单（人类可读版 + JSON Schema 导出）
# ==============================================================================
if __name__ == "__main__":
    import json
    import os

    # 输出到脚本所在目录，避免硬编码绝对路径导致跨机器失效
    OUTPUT_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "4_10_7_2_tools_output.json",
    )
    SEP = "=" * 72
    print(SEP)
    print(" 自动代码审查 Agent — 工具清单（共 4 个）")
    print(SEP)

    total = len(CODE_REVIEW_TOOLS)
    for idx, tool in enumerate(CODE_REVIEW_TOOLS, 1):
        print()
        print(f"【Tool {idx}/{total}】 {tool['name']}")
        print(f"  安全级别 : {tool['security_level'].value}")
        print(f"  安全说明 : {tool['security_reason']}")
        desc = tool['description']
        if len(desc) > 160:
            desc = desc[:160] + "……"
        print(f"  功能描述 : {desc}")
        req = tool["params"].get("required", [])
        opt = [k for k in tool["params"].get("properties", {}) if k not in req]
        print(f"  必填参数 : {', '.join(req)}")
        print(f"  可选参数 : {', '.join(opt)}")

    print()
    print(SEP)
    print(" JSON Schema 完整导出")
    print(SEP)

    # 转为可序列化结构（处理 Enum）
    serializable = []
    for t in CODE_REVIEW_TOOLS:
        t_copy = dict(t)
        t_copy["security_level"] = t["security_level"].value
        serializable.append(t_copy)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"tools": serializable}, f, ensure_ascii=False, indent=2)

    print(f"✅ 完整 JSON Schema 已写入: {OUTPUT_PATH}")
    print(f"   包含字段: name / description / security_level / security_reason")
    print(f"           / params (标准 JSON Schema draft-07) / returns_example")

    openai_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "4_10_7_2_tools_openai.json",
    )
    with open(openai_path, "w", encoding="utf-8") as f:
        json.dump(to_openai_function_specs(), f, ensure_ascii=False, indent=2)
    print(f"✅ OpenAI Function Calling 格式已写入: {openai_path}")