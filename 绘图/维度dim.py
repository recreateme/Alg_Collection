import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12})  # 设置学术字体
plt.rcParams['font.family'] = 'SimHei'

# 生成模拟数据（带可重复性的随机种子）
np.random.seed(42)
dim = [8,16,32,64,128]
recall_scores = np.random.rand(len(dim))
ndcg_scores = np.random.rand(len(dim))

# 创建画布与坐标轴
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.grid(True, linestyle='--', alpha=0.6)  # 添加网格线

# 绘制Recall曲线（左轴）
ax1.set_xlabel('嵌入维度 $\mathcal{d}$', fontweight='bold')
ax1.set_ylabel('Recall@20', color='#2C5F8A', fontweight='bold')
line1, = ax1.plot(dim, recall_scores,
                 color='#2C5F8A', marker='o',
                 markersize=8, linewidth=2,
                 linestyle='-', label='Recall@20')

# 设置左轴刻度参数
ax1.tick_params(axis='y', labelcolor='#2C5F8A')
ax1.set_xticks(dim)  # 确保X轴显示整数

# 创建右轴绘制NDCG曲线
ax2 = ax1.twinx()
ax2.set_ylabel('NDCG@20', color='#E84A22', fontweight='bold')
line2, = ax2.plot(dim, ndcg_scores,
                 color='#E84A22', marker='s',
                 markersize=8, linewidth=2,
                 linestyle='--', label='NDCG@20')

# 设置右轴刻度参数
ax2.tick_params(axis='y', labelcolor='#E84A22')

# 合并图例
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels,
          loc='upper center',
          bbox_to_anchor=(0.5, 1.15),
          ncol=2, frameon=False)

# 添加分析标注
# ax1.text(6, 0.52, 'Optimal Layer=3',
#         fontstyle='italic', color='#333333',
#         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 图表美化
plt.tight_layout()

# 保存矢量图格式（适合论文投稿）
# plt.savefig('gnn_layer_ablation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('ssl_ablation.png', transparent=True, bbox_inches='tight', dpi=300)
plt.show()