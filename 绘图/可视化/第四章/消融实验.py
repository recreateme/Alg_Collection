import matplotlib.pyplot as plt
import numpy as np

# 数据准备（四个模型）
models = ['DCNS-bi', 'DCNS-sample', 'DCNS-pac', 'DCNS']
ml1m_auc = [0.909, 0.916, 0.919, 0.930]
ml1m_f1 = [0.828, 0.841, 0.840, 0.846]
book_auc = [0.780, 0.779, 0.788, 0.795]
book_f1 = [0.695, 0.696, 0.699, 0.704]

# 设置柱状图位置
n_models = len(models)
width = 0.15  # 减小柱宽
x = np.array([0, 1])  # 横轴位置：0=AUC, 1=F1
offsets = [-(width*1.5), -width/2, width/2, width*1.5]  # 四个模型的偏移量

# 定义灰度阴影和图案
grays = ['0.8', '0.6', '0.4', '0.2']  # 灰度从浅到深 (0=黑, 1=白)
hatches = ['/', '\\', '//', '||']     # 不同方向的线条：斜线、反斜线、密集斜线、竖线

# 创建子图，调整figure大小
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

fontsize = 14  # 所有字体大小设置为14

# 子图1: ML-1M
for i, (model, offset, gray, hatch) in enumerate(zip(models, offsets, grays, hatches)):
    bar1 = ax1.bar(x[0] + offset, ml1m_auc[i], width, label=model, color=gray)
    bar2 = ax1.bar(x[1] + offset, ml1m_f1[i], width, color=gray)
    # 添加图案
    bar1[0].set_hatch(hatch)
    bar2[0].set_hatch(hatch)

# 设置标题和横轴
ax1.set_title('ML-1M', fontsize=fontsize)
ax1.set_xticks(x)
ax1.set_xticklabels(['AUC', 'F1'], fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=fontsize)  # 图例在右上角

# 在柱状图上方标注数值（100倍）
for i, (v, offset) in enumerate(zip(ml1m_auc, offsets)):
    ax1.text(x[0] + offset, v + 0.005, f'{v*100:.1f}', ha='center', va='bottom', fontsize=fontsize)
for i, (v, offset) in enumerate(zip(ml1m_f1, offsets)):
    ax1.text(x[1] + offset, v + 0.005, f'{v*100:.1f}', ha='center', va='bottom', fontsize=fontsize)

# 调整纵轴范围，放大差异
ax1.set_ylim(0.8, 0.95)

# 移除纵轴刻度和标签
ax1.set_yticks([])
ax1.set_ylabel('', fontsize=fontsize)

# 子图2: Book-Crossing
for i, (model, offset, gray, hatch) in enumerate(zip(models, offsets, grays, hatches)):
    bar1 = ax2.bar(x[0] + offset, book_auc[i], width, label=model, color=gray)
    bar2 = ax2.bar(x[1] + offset, book_f1[i], width, color=gray)
    # 添加图案
    bar1[0].set_hatch(hatch)
    bar2[0].set_hatch(hatch)

# 设置标题和横轴
ax2.set_title('Book-Crossing', fontsize=fontsize)
ax2.set_xticks(x)
ax2.set_xticklabels(['AUC', 'F1'], fontsize=fontsize)
ax2.legend(loc='upper right', fontsize=fontsize)  # 图例在右上角

# 在柱状图上方标注数值（100倍）
for i, (v, offset) in enumerate(zip(book_auc, offsets)):
    ax2.text(x[0] + offset, v + 0.005, f'{v*100:.1f}', ha='center', va='bottom', fontsize=fontsize)
for i, (v, offset) in enumerate(zip(book_f1, offsets)):
    ax2.text(x[1] + offset, v + 0.005, f'{v*100:.1f}', ha='center', va='bottom', fontsize=fontsize)

# 调整纵轴范围，放大差异
ax2.set_ylim(0.65, 0.82)

# 移除纵轴刻度和标签
ax2.set_yticks([])
ax2.set_ylabel('', fontsize=fontsize)

# 调整布局
plt.tight_layout()

# 保存图形到当前文件夹，使用高分辨率
plt.savefig('comparison_chart.png', dpi=300)

# 显示图形
plt.show()