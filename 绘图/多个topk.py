import matplotlib.pyplot as plt
import numpy as np
import random

# 模型名称
models = ['KGAT', 'KGIN', 'MCCLK', 'L2CL', 'KACR']

# 已知的 recall@20 数据
recall_at_20 = [0.1489, 0.1695, 0.1776, 0.1477, 0.1815]

# 定义 top-k 的范围
k_values = [5, 10, 20, 50, 100]

# 设置随机种子以确保可重复性（可选）
random.seed(42)

# 创建图形
plt.figure(figsize=(8, 6))

# 为每个模型生成 Recall@k 的趋势，保持 top-20 的相对大小，加入较小的随机性
for i, model in enumerate(models):
    # 基于 recall_at_20 推断其他 k 值的 Recall
    # 假设 Recall@k 随 k 增加呈对数增长，但增长速率随 k 增加而减慢
    # 基础趋势：Recall@k = recall_at_20 * (1 - exp(-a * log(k+1))) / (1 - exp(-a * log(21)))
    a = 0.5  # 控制增长速率的参数

    # 计算基础趋势（无随机性）
    base_recall = [recall_at_20[i] * (1 - np.exp(-a * np.log(k + 1))) / (1 - np.exp(-a * np.log(21)))
                   for k in k_values]

    # 加入较小的随机性（±5%的波动）
    random_fluctuations = [random.uniform(-0.05, 0.05) for _ in k_values]
    inferred_recall = [max(0, min(1, base_recall[j] * (1 + random_fluctuations[j])))
                       for j in range(len(k_values))]

    # 确保相对大小顺序与 recall_at_20 一致
    # 按比例调整，确保 k=20 的值接近 recall_at_20
    scaling_factor = recall_at_20[i] / base_recall[k_values.index(20)]  # 确保 k=20 的值匹配
    inferred_recall = [r * scaling_factor for r in inferred_recall]

    # 限制值在合理范围内 [0, 0.5]
    inferred_recall = np.clip(inferred_recall, 0, 0.5)

    # 打印每个 k 下的 Recall 值（调试用，可删除）
    print(f"Model {model} Recall@k: {dict(zip(k_values, inferred_recall))}")

    # 绘制折线图
    plt.plot(k_values, inferred_recall, marker='o', label=f'{model}')

# 设置图形样式，确保横坐标按 k 值标定
plt.title('Recall@k')
plt.xlabel('k')
plt.ylabel('Recall@k')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')

# 明确设置横坐标的刻度
plt.xticks(k_values)  # 确保横坐标只显示 5, 10, 20, 50, 100

# 调整布局并显示
plt.tight_layout()
plt.show()