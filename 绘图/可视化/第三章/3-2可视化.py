import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ['KGAT', 'KGIN', 'MCCLK', 'L2CL', 'KACR']

# 第二组数据
recall_at_20 = [0.0870, 0.0956, 0.1009, 0.0862, 0.1056]
ndcg_at_20 = [0.1325, 0.1408, 0.1448, 0.1310, 0.1490]

# 设置柱状图的位置
x = np.arange(len(models))  # 模型的标签位置
width = 0.35  # 柱状图的宽度

# 创建图形和主纵坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 统一刻度范围
all_values = recall_at_20 + ndcg_at_20
min_value = min(all_values)
max_value = max(all_values)
ax1.set_ylim(min_value - 0.01, max_value + 0.01)  # 设置统一的刻度范围

# 绘制 recall@20 的柱状图（左侧纵坐标）
rects1 = ax1.bar(x - width/2, recall_at_20, width, label='Recall@20', color='#555555', edgecolor='black')  # 深灰色

# 设置左侧纵坐标轴标签
ax1.set_ylabel('Recall@20 (%)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelsize=12, labelcolor='black')

# 创建右侧纵坐标轴
ax2 = ax1.twinx()
ax2.set_ylim(min_value - 0.01, max_value + 0.01)  # 设置统一的刻度范围

# 绘制 ndcg@20 的柱状图（右侧纵坐标）
rects2 = ax2.bar(x + width/2, ndcg_at_20, width, label='NDCG@20', color='#AAAAAA', edgecolor='black')  # 浅灰色

# 设置右侧纵坐标轴标签
ax2.set_ylabel('NDCG@20 (%)', fontsize=12, color='black')
ax2.tick_params(axis='y', labelsize=12, labelcolor='black')

# 设置横坐标轴标签
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)

# 添加图例，使其靠近上边框
fig.legend(loc='upper center', bbox_to_anchor=(0.25, 0.95), ncol=2, fontsize=12)  # ncol=2 使图例水平排列

# 添加横向的网格线（仅左侧纵坐标）
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 在每个柱状图上显示数值
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height*100:.2f}%',  # 转换为百分比形式
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

autolabel(rects1, ax1)
autolabel(rects2, ax2)

# 调整布局
fig.tight_layout()

# 显示图形
plt.show()