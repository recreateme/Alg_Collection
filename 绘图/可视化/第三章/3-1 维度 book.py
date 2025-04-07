import matplotlib.pyplot as plt
import numpy as np

# 维度列表
dims = [8, 16, 32, 64, 128]

# Amazon-Book 数据
recall_amazon = [0.1238, 0.1358, 0.1520, 0.1815, 0.1804]  # Recall@20
ndcg_amazon = [0.0800, 0.0900, 0.1100, 0.1189, 0.1130]  # NDCG@20

# 创建均匀分布的横轴（5 个点，对应 5 个维度）
x_uniform = np.arange(len(dims))  # 均匀分布的横轴（0, 1, 2, 3, 4）

# 创建一个图形（用于 Amazon-Book）
fig, ax1 = plt.subplots(figsize=(8, 6))

# Amazon-Book 折线图
ax1.plot(x_uniform, recall_amazon, marker='o', label='Recall@20', color='skyblue',
         linestyle='--', linewidth=2)  # 虚线表示 Recall
ax1.plot(x_uniform, ndcg_amazon, marker='s', label='NDCG@20', color='salmon',
         linestyle='-', linewidth=2)  # 实线表示 NDCG

# 设置 Amazon-Book 图表的横轴和纵轴
ax1.set_xlabel('dim', fontsize=14)
ax1.set_ylabel('Recall', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax1.set_xticks(x_uniform)  # 设置均匀分布的刻度
ax1.set_xticklabels(dims)  # 显示实际维度值

ax1.grid(True, linestyle='--', alpha=0.7)

# 添加右侧纵轴（Recall）给 Amazon-Book
ax1_r = ax1.twinx()
ax1_r.set_ylabel('Recall@20', fontsize=14)
ax1_r.tick_params(axis='y', labelsize=11)
ax1_r.set_ylim(0, max(recall_amazon) * 1.2)  # 调整 Recall 纵轴范围

# 为图表添加图例（放在内部右侧中间，垂直排列）
ax1.legend(fontsize=12)
# 添加数值标签（在折线上每个点上）
# for j, dim in enumerate(x_uniform):
    # Recall 标签（默认黑色，虚线对应的值）
    # ax1.text(dim, recall_amazon[j] + 0.005, f'{recall_amazon[j]*100:.2f}%',
    #         ha='center', va='bottom', fontsize=10)
    # NDCG 标签（默认黑色，实线对应的值）
    # ax1.text(dim, ndcg_amazon[j] + 0.005, f'{ndcg_amazon[j]*100:.2f}%',
    #         ha='center', va='bottom', fontsize=10)

# 调整布局并显示
plt.tight_layout()
plt.show()