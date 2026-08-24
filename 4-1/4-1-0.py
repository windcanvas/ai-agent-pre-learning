# 用代码直观感受"预测下一个词"
# 假设我们有一个非常简化的"模型"——只记住了一些词组
simple_model = {
    "牛肉": ["面", "饭", "汤"],
    "天气": ["不错", "很好", "太热了"],
    "1+1=": ["2"],
    "hello": ["world", "there", "!"],
}

def naive_predict(text):
    """最简单的"预测下一个词"——查表"""
    last_word = text.split()[-1] if text.split() else ""
    candidates = simple_model.get(last_word, ["???"])
    return candidates[0]  # 实际 LLM 会根据概率挑选，不是硬编码

print(naive_predict("老板来碗 牛肉"))   # 面
print(naive_predict("今天 天气"))      # 不错
print(naive_predict("1+1="))          # 2
