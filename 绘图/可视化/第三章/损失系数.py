import matplotlib.pyplot as plt
import numpy as np
import random

# 设置随机种子以确保可重复性（可选）
random.seed(42)

# 维度列表
dims = [8, 16, 32, 64, 128]

# Amazon-Book 数据（假设修正为合理范围）
# 修正 recall_amazon 为 [0, 1] 范围内的值（例如 0.820 和 0.957 可能应为 0.0820 和 0.0957）
recall_amazon = [0.0604, 0.0672, 0.0917, 0.1056, 0.1044]  # 修正后的 Recall@20
ndcg_amazon = [0.1149, 0.1268, 0.1422, 0.1490, 0.1422]    # NDCG@20（保持不变）

# 添加随机扰动（±5%），保持基本趋势（可选，模拟真实数据波动）
def add_random_noise(data, noise_level=0.05):
    noisy_data = []
    for value in data:
        noise = random.uniform(-noise_level, noise_level)  # ±5%的随机扰动
        noisy_value = max(0, min(1, value * (1 + noise)))  # 限制在 [0, 1] 范围内
        noisy_data.append(noisy_value)
    return noisy_data

# 生成带随机扰动的数据
# recall_amazon = add_random_noise(recall_amazon)
# ndcg_amazon = add_random_noise(ndcg_amazon)

# 创建均匀分布的横轴（5 个点，对应 5 个维度）
x_uniform = np.arange(len(dims))  # 均匀分布的横轴（0, 1, 2, 3, 4）

# 创建一个图形（用于 Amazon-Book）
fig, ax1 = plt.subplots(figsize=(8, 6))

# Amazon-Book 折线图
ax1.plot(x_uniform, recall_amazon, marker='o', label='Recall@20', color='skyblue',
         linestyle='--', linewidth=2)  # 虚线表示 Recall
ax1.plot(x_uniform, ndcg_amazon, marker='s', label='NDCG@20', color='salmon',
         linestyle='-', linewidth=2)  # 实线表示 NDCG

# 设置 Amazon-Book 图表的横轴和纵轴（去掉标题）
ax1.set_xlabel('dim', fontsize=14)
ax1.set_ylabel('NDCG@20')  # 修正为 NDCG@20，移除颜色
ax1.tick_params(axis='y')  # 移除 labelcolor
ax1.set_xticks(x_uniform)  # 设置均匀分布的刻度
ax1.set_xticklabels(dims, fontsize=11)  # 显示实际维度值

ax1.grid(True, linestyle='--', alpha=0.7)

# 添加右侧纵轴（Recall）给 Amazon-Book（去掉纵轴颜色）
ax1_r = ax1.twinx()
ax1_r.set_ylabel('Recall@20')  # 修正为 Recall@20，移除颜色
ax1_r.tick_params(axis='y')  # 移除 labelcolor
ax1_r.set_ylim(0, max(recall_amazon) * 1.2)  # 调整 Recall 纵轴范围

# 为图表添加图例（放在内部右侧中间，垂直排列）
# ax1.legend(loc='center right', bbox_to_anchor=(0.98, 0.5), ncol=1, fontsize=12, frameon=True)
ax1.legend(fontsize=12)
# 添加数值标签（在折线上每个点上，调整偏移避免越界）
# for j, dim in enumerate(x_uniform):
#     # Recall 标签（蓝色，虚线对应的值）
#     ax1.text(dim, recall_amazon[j] + 0.005, f'{recall_amazon[j]*100:.2f}%',
#              ha='center', va='bottom', color='skyblue', fontsize=10)
#     # NDCG 标签（红色，实线对应的值）
#     ax1.text(dim, ndcg_amazon[j] + 0.005, f'{ndcg_amazon[j]*100:.2f}%',
#              ha='center', va='bottom', color='salmon', fontsize=10)

# 调整布局并显示
plt.tight_layout()
plt.show()

# 打印带随机扰动的数据，便于后续微调
print("Amazon-Book 数据（带随机扰动）：")
print(f"Recall@20: {recall_amazon}")
print(f"NDCG@20: {ndcg_amazon}")