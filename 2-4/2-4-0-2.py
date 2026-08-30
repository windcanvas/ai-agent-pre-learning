import os

from dotenv import load_dotenv

load_dotenv()  # 启动时自动把项目根目录的 .env 注入环境变量

import httpx
import json
import time

# ✅ 流式 = 服务端逐字发（stream: True） + 客户端逐块收（httpx.stream）
start = time.time()

with httpx.stream(
    "POST",
    "https://api.deepseek.com/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
    json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "写一篇 1000 字的 AI Agent 介绍"}],
        "stream": True,  # ← 流式：一个字一个字返回
    },
    timeout=120,
) as response:
    # 逐行读取返回的 SSE 事件流
    for line in response.iter_lines():
        if line.startswith("data: "):
            data_str = line[6:]  # 去掉 "data: " 前缀
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta:
                print(delta, end="", flush=True)   # ← 实时打印，一个字一个字出来

print()
print(f"\n流式接收完毕，总耗时 {time.time() - start:.1f} 秒")
