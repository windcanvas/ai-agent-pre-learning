# 用代码直观感受 Temperature 的效果
import math

def softmax_with_temperature(logits, temperature=1.0):
    """Temperature 如何影响选词概率"""
    # logits 可以理解为模型对每个候选词的"信心分数"
    # Temperature 越低 → 高分词的优势越被放大 → 输出越确定
    scaled = [l / temperature for l in logits]
    max_scaled = max(scaled)
    exp_sum = sum(math.exp(s - max_scaled) for s in scaled)
    probs = [math.exp(s - max_scaled) / exp_sum for s in scaled]
    return probs

# 假设模型对 5 个候选词的信心分数
logits = [5.0, 3.0, 1.0, 0.5, 0.1]

for temp in [0.2, 0.5, 1.0, 1.5]:
    probs = softmax_with_temperature(logits, temp)
    print(f"Temperature={temp}: 第一名概率 ={probs[0]:.0%}, 第二名 ={probs[1]:.0%}, 第三名 ={probs[2]:.0%}")
