import os

from dotenv import load_dotenv

load_dotenv()  # 启动时自动把项目根目录的 .env 注入环境变量

import httpx
import time

# 调用 LLM API（非流式）
start = time.time()

response = httpx.post(
    "https://api.deepseek.com/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
    json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "写一篇 1000 字的 AI Agent 介绍"}],
        "stream": False,  # ← 非流式：等全部生成完才返回
    },
    timeout=120,
)

data = response.json()
content = data["choices"][0]["message"]["content"]
print(f"等了 {time.time() - start:.1f} 秒，终于收到完整回复：")
print(content)
