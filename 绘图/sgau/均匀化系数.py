import matplotlib.pyplot as plt
import numpy as np

# 维度列表
tau = [0.1, 0.2, 0.4, 0.6, 0.8]

# Amazon-Book 数据
recall_amazon = [0.846, 0.858, 0.895, 0.930, 0.925]  # Recall@20
ndcg_amazon = [0.766, 0.814, 0.830, 0.844, 0.846]  # NDCG@20

# 创建均匀分布的横轴（5 个点，对应 5 个维度）
x_uniform = np.arange(len(tau))  # 均匀分布的横轴（0, 1, 2, 3, 4）

# 创建一个图形（用于 Amazon-Book）
fig, ax1 = plt.subplots(figsize=(8, 6))

# Amazon-Book 折线图
ax1.plot(x_uniform, recall_amazon, marker='o', label='Recall@2/XMLSchemaDoc', color='skyblue',
         linestyle='--', linewidth=2)  # 虚线表示 Recall
ax1.plot(x_uniform, ndcg_amazon, marker='s', label='NDCG@20', color='salmon',
         linestyle='-', linewidth=2)  # 实线表示 NDCG

# 设置 Amazon-Book 图表的横轴和纵轴
ax1.set_xlabel(r'$\rho$', fontsize=14)
ax1.set_ylabel('Recall@20', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax1.set_xticks(x_uniform)  # 设置均匀分布的刻度
ax1.set_xticklabels(tau)  # 显示实际维度值

ax1.grid(True, linestyle='--', alpha=0.7)

# 添加右侧纵轴（NDCG）给 Amazon-Book
ax2 = ax1.twinx()
ax2.set_ylabel('NDCG@20', fontsize=14)
ax2.tick_params(axis='y', labelsize=11)
ax2.set_ylim(0, max(ndcg_amazon) * 1.2)  # 调整 NDCG 纵轴范围

# 为图表添加图例（放在内部右侧中间，垂直排列）
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2, fontsize=12)

# 调整布局并显示
plt.tight_layout()
plt.show()