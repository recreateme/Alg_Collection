import matplotlib.pyplot as plt
import numpy as np

# 数据
x = [0.01, 0.1, 0.2, 0.5, 1, 2, 5]
y_left = [0.0937, 0.1015, 0.1288, 0.1397, 0.1366, 0.1304, 0.1245]
y_right = [0.0952, 0.1002, 0.1019, 0.1063, 0.1047, 0.1011, 0.0952]

# 创建画布
fig, ax1 = plt.subplots(figsize=(10, 8))

# 绘制左侧y轴的数据
ax1.plot(x, y_left, label='Left Y-Axis Data', color='blue', marker='o', linestyle='-', linewidth=2)
ax1.scatter(x, y_left, label='Left Y-Axis Points', color='blue', marker='o', s=100)

# 设置左侧y轴的属性
ax1.set_xlabel(r'$\lambda_1$', fontsize=14)  # 使用LaTeX样式设置x轴标签
ax1.set_ylabel('Recall@20')
ax1.tick_params(axis='y')
ax1.set_ylim(min(y_left) * 0.95, max(y_left) * 1.05)

# 设置x轴刻度为整数
ax1.set_xticks([1, 2, 3, 4, 5])  # 只显示整数刻度

# 创建共享x轴的第二个y轴
ax2 = ax1.twinx()

# 绘制右侧y轴的数据
ax2.plot(x, y_right, label='Recall@20', color='red', marker='^', linestyle='-', linewidth=2)
ax2.scatter(x, y_right, label='NDCG@20', color='red', marker='^', s=100)

# 设置右侧y轴的属性
ax2.set_ylabel('NDCG@20')
ax2.tick_params(axis='y')
ax2.set_ylim(min(y_right) * 0.95, max(y_right) * 1.05)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=14)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
