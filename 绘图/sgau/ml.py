import matplotlib.pyplot as plt
import numpy as np

# 第一个数据集
x1 = [0.01, 0.1, 0.2, 0.5, 1, 2, 5]
y_left1 = [0.0937, 0.1015, 0.1288, 0.1397, 0.1366, 0.1304, 0.1245]
y_right1 = [0.0952, 0.1002, 0.1019, 0.1063, 0.1047, 0.1024, 0.1000]

# 第二个数据集
x2 = [0.01, 0.1, 0.2, 0.5, 1, 2, 5]
y_left2 = [0.1804, 0.1934, 0.2115, 0.2166, 0.2140, 0.2001, 0.1867]  # Recall 数据
y_right2 = [0.1209, 0.1415, 0.1467, 0.1507, 0.1488, 0.1455, 0.139]  # NDCG 数据

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 绘制第一个子图
ax1.set_title('Amazon-Book', fontsize=16)
ax1.plot(x1, y_left1, label='Recall@20', color='blue', marker='o', linestyle='-', linewidth=2)
ax1.scatter(x1, y_left1, color='blue', s=100)
ax1.set_xlabel(r'$\lambda_1$', fontsize=14)
ax1.set_ylabel('Recall@20', fontsize=14)
ax1.tick_params(axis='y')
ax1.set_ylim(min(y_left1) * 0.95, max(y_left1) * 1.05)

ax1_twin = ax1.twinx()
ax1_twin.plot(x1, y_right1, label='NDCG@20', color='red', marker='^', linestyle='-', linewidth=2)
ax1_twin.scatter(x1, y_right1, color='red', s=100)
ax1_twin.set_ylabel('NDCG@20', fontsize=14)
ax1_twin.tick_params(axis='y')

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=14)

# 绘制第二个子图
ax2.set_title('ML-20M', fontsize=16)
ax2.plot(x2, y_left2, label='Recall@20', color='cyan', marker='o', linestyle='-', linewidth=2)
ax2.scatter(x2, y_left2, color='cyan', s=100)
ax2.set_xlabel(r'$\lambda_1$', fontsize=14)
ax2.set_ylabel('Recall@20', fontsize=14)
ax2.tick_params(axis='y')
ax2.set_ylim(min(y_left2) * 0.95, max(y_left2) * 1.05)

ax2_twin = ax2.twinx()
ax2_twin.plot(x2, y_right2, label='NDCG@20', color='orange', marker='^', linestyle='-', linewidth=2)
ax2_twin.scatter(x2, y_right2, color='orange', s=100)
ax2_twin.set_ylabel('NDCG@20', fontsize=14)
ax2_twin.tick_params(axis='y')

# 添加图例
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=14)

# 调整布局
plt.tight_layout()
plt.savefig('ml.png', dpi=600)
# 显示图表
plt.show()
