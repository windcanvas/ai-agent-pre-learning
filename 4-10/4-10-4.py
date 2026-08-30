"""Agent 核心循环 —— LLM + Context + Tool 三者的工程整合"""
import json
from datetime import datetime
from openai import OpenAI


class Agent:
    """一个完整的 Agent 实现，展示 LLM/Context/Tool 的实际协作"""

    def __init__(self, name: str, llm_config: dict, registry: ToolRegistry):
        self.name = name
        self.llm = OpenAI(**llm_config)
        self.registry = registry
        self.model = llm_config.get("default_model", "deepseek-chat")

        # Context 组件
        self.system_prompt = ""
        self.history: list[dict] = []
        self.memory: dict = {}

    def configure(self, system_prompt: str, tools_category: str = None):
        """配置 Agent——设定行为规则和可用工具"""
        tool_list = self.registry.get_for_llm(tools_category)
        self.system_prompt = f"""{system_prompt}

## 可用工具
{json.dumps(tool_list, ensure_ascii=False, indent=2)}

## 调用格式
需要工具时输出:
```json
{{"tool":"工具名","params":{{"参数":"值"}}}}
{{"done":true,"summary":"完成摘要"}}
```"""

    def run(self, task: str, max_steps: int = 10) -> dict:
        """执行任务——返回完整的执行轨迹"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            # 注入记忆
            {"role": "system", "content": f"[记忆]\n{json.dumps(self.memory, ensure_ascii=False)}"},
            {"role": "user", "content": task},
        ]

        trace = []  # 记录每一步做了什么——调试用

        for step in range(max_steps):
            # ===== LLM 推理 =====
            resp = self.llm.chat.completions.create(
                model=self.model, messages=messages, temperature=0,
            )
            content = resp.choices[0].message.content

            # ===== 尝试解析工具调用或完成信号 =====
            tool_call = self._parse_response(content)

            if tool_call["type"] == "done":
                trace.append({"step": step, "action": "done", "result": tool_call["summary"]})
                return {
                    "status": "ok",
                    "result": tool_call["summary"],
                    "steps": step + 1,
                    "trace": trace,
                }

            if tool_call["type"] == "tool":
                tool_name = tool_call["name"]
                params = tool_call["params"]

                # ===== Token 预算检查 =====
                if self._token_count(messages) > 100000:
                    # 上下文快满了 → 压缩
                    messages = self._compress_history(messages)

                # ===== 工具执行（通过 Registry 统一管理） =====
                result = self.registry.execute(tool_name, params)
                trace.append({
                    "step": step, "tool": tool_name,
                    "params": params, "status": result["status"],
                })

                # ===== 结果回传（更新 Context） =====
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                })

                # ===== 重复检测 =====
                if self._is_stuck(trace):
                    messages.append({
                        "role": "system",
                        "content": "⚠️ 检测到连续重复操作。请换一种策略或向用户说明当前困境。",
                    })

        return {
            "status": "max_steps_reached",
            "steps": max_steps,
            "trace": trace,
        }

    def _parse_response(self, content: str) -> dict:
        """解析 LLM 输出——工具调用 or 完成"""
        try:
            # 提取 JSON 块
            if "```json" in content:
                start = content.index("```json") + 7
                end = content.index("```", start)
                data = json.loads(content[start:end])
            else:
                data = json.loads(content)

            if "done" in data:
                return {"type": "done", "summary": data.get("summary", content)}
            if "tool" in data:
                return {"type": "tool", "name": data["tool"], "params": data.get("params", {})}
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return {"type": "unknown", "content": content}

    def _token_count(self, messages: list[dict]) -> int:
        """粗略 Token 计数"""
        return sum(len(m["content"]) // 2 for m in messages)

    def _compress_history(self, messages: list[dict]) -> list[dict]:
        """压缩长对话——保留 System + 最近几轮"""
        system_msgs = [m for m in messages if m["role"] == "system"]
        recent = messages[-8:]  # 保留最近 8 条
        return system_msgs + recent

    def _is_stuck(self, trace: list[dict]) -> bool:
        """检测是否陷入重复循环"""
        if len(trace) < 3:
            return False
        recent = trace[-3:]
        return all(
            t.get("tool") == recent[0].get("tool")
            and t.get("params") == recent[0].get("params")
            for t in recent
        )
