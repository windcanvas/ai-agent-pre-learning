# pip install tiktoken
import tiktoken

# OpenAI 的编码器
enc = tiktoken.encoding_for_model("gpt-4o")

def show_tokens(text: str):
    tokens = enc.encode(text)
    print(f"文本: {text}")
    print(f"Token 数量: {len(tokens)}")
    print(f"Token 列表: {tokens}")
    print(f"逐个解码: {[enc.decode([t]) for t in tokens]}")
    print()

show_tokens("Hello, world!")
# 文本: Hello, world!
# Token 数量: 4
# 逐个解码: ['Hello', ',', ' world', '!']

show_tokens("人工智能改变世界")
# 文本: 人工智能改变世界
# Token 数量: 4
# 逐个解码: ['人工', '智能', '改变', '世界']

show_tokens("1+1=2")
# Token 数量: 5
# 逐个解码: ['1', '+', '1', '=', '2']

# 看看 1000 Token 大约是多少中文字
text = "大语言模型是人工智能领域的重要突破。" * 50
tokens = len(enc.encode(text))
print(f"{len(text)} 个中文字 ≈ {tokens} 个 Token")
# 约 800 个中文字 ≈ 1000 个 Token
