import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ['KGAT', 'KGIN', 'MCCLK', 'L2CL', 'KACR']

# 新数据
recall = [0.1489, 0.1695, 0.1746, 0.1477, 0.1815]
ndcg = [0.1006, 0.1102, 0.1152, 0.0980, 0.1189]

# 设置柱状图的位置
x = np.arange(len(models))  # 模型的标签位置
width = 0.35  # 柱状图的宽度

# 创建图形和主纵坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 统一刻度范围
all_values = recall + ndcg
min_value = min(all_values)
max_value = max(all_values)
ax1.set_ylim(min_value - 0.01, max_value + 0.01)  # 设置统一的刻度范围

# 绘制 recall 的柱状图（左侧纵坐标）
rects1 = ax1.bar(x - width/2, recall, width, label='Recall@20', color='#555555', edgecolor='black')  # 深灰色

# 设置左侧纵坐标轴标签
ax1.set_ylabel('Recall', color='black', fontsize=14)  # 调整字号
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)  # 调整刻度字号

# 创建右侧纵坐标轴
ax2 = ax1.twinx()
ax2.set_ylim(min_value - 0.01, max_value + 0.01)  # 设置统一的刻度范围

# 绘制 ndcg 的柱状图（右侧纵坐标）
rects2 = ax2.bar(x + width/2, ndcg, width, label='NDCG@20', color='#AAAAAA', edgecolor='black')  # 浅灰色

# 设置右侧纵坐标轴标签
ax2.set_ylabel('NDCG', color='black', fontsize=14)  # 调整字号
ax2.tick_params(axis='y', labelcolor='black', labelsize=12)  # 调整刻度字号

# 设置横坐标轴标签
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)  # 调整横坐标标签字号

# 添加图例，使其靠近上边框
fig.legend(loc='upper center', bbox_to_anchor=(0.25, 0.95), ncol=2, fontsize=12)  # 调整图例字号

# 添加横向的网格线（仅左侧纵坐标）
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 在每个柱状图上显示数值（调整为百分数）
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        # 将数值乘以 100 并格式化为百分数
        ax.annotate('{:.2f}%'.format(height * 100),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)  # 调整标注字号

autolabel(rects1, ax1)
autolabel(rects2, ax2)

# 调整布局
fig.tight_layout()

# 显示图形
plt.show()