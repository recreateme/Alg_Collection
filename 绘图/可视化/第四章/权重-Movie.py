import matplotlib.pyplot as plt
import numpy as np

# 维度列表
tau = [0.5, 0.6, 0.7, 0.8, 0.9]


# 数据
AUC = [0.718, 0.722, 0.740, 0.795, 0.792]  # AUC 数据，峰值在 tau=0.7
F1 = [0.680, 0.692, 0.698, 0.700, 0.704]    # F1 数据，峰值在 tau=0.8

# 创建均匀分布的横轴
x_uniform = np.arange(len(tau))  # 均匀分布的横轴（0, 1, 2, 3, 4）

# 创建图形
fig, ax1 = plt.subplots(figsize=(8, 6))

# 定义颜色
auc_color = 'blue'
f1_color = 'red'

# 左侧 y 轴 (AUC)
ax1.plot(x_uniform, AUC, marker='o', label='AUC', color=auc_color,
         linestyle='--', linewidth=2)  # 虚线表示 AUC
ax1.set_xlabel(r'($\rho$)', fontsize=15)
ax1.set_ylabel('AUC', fontsize=14)
ax1.tick_params(axis='y', labelsize=11, colors=auc_color)  # 左侧刻度颜色与 AUC 一致
ax1.tick_params(axis='x', labelsize=11)
ax1.set_xticks(x_uniform)
ax1.set_xticklabels(tau)

# 只保留水平网格线
ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)

# 设置 y 轴范围为 [0.75, 0.95]
ax1.set_ylim(0.7, 0.8)

# 右侧 y 轴 (F1)
ax2 = ax1.twinx()
ax2.plot(x_uniform, F1, marker='s', label='F1', color=f1_color,
         linestyle='-', linewidth=2)  # 实线表示 F1
ax2.set_ylabel('F1', fontsize=14)
ax2.tick_params(axis='y', labelsize=11, colors=f1_color)  # 右侧刻度颜色与 F1 一致

# 设置右侧 y 轴范围为 [0.75, 0.95]
ax2.set_ylim(0.65, 0.72)

# 添加图例（合并两条线的图例）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

# 调整布局并显示
plt.tight_layout()
plt.show()
