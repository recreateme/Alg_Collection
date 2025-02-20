import matplotlib.pyplot as plt
import numpy as np

# Amazon-Book 数据
groups_amazon = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']
sgl_amazon = 0.1399
sgau_amazon = 0.1408
ratio1_amazon = np.array([0.005, 0.015, 0.09, 0.25, 0.64])
ratio2_amazon = np.array([0.015, 0.055, 0.065, 0.29, 0.61])
ratio1_amazon = ratio1_amazon / np.sum(ratio1_amazon)
ratio2_amazon = ratio2_amazon / np.sum(ratio2_amazon)
sgl_recall_amazon = ratio1_amazon * sgl_amazon
sgau_recall_amazon = ratio2_amazon * sgau_amazon

# Movielens 数据
groups_movielens = ['group 1', 'group 2', 'group 3', 'group 4', 'group 5']
sgl_movielens = 0.2152
sgau_movielens = 0.2219
ratio1_movielens = np.array([0.006, 0.03, 0.11, 0.22, 0.55])
ratio1_movielens = ratio1_movielens / np.sum(ratio1_movielens)

ratio2_movielens = np.array([0.01, 0.03, 0.14, 0.18, 0.40])
ratio2_movielens = ratio2_movielens / np.sum(ratio2_movielens)

sgl_recall_movielens = ratio1_movielens * sgl_movielens
sgau_recall_movielens = ratio2_movielens * sgau_movielens

# 设置图形和子图
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# 设置柱状图的宽度和位置
x = np.arange(len(groups_amazon))
bar_width = 0.35

# 绘制 Amazon-Book 数据
axs[0].bar(x - bar_width/2, sgl_recall_amazon, width=bar_width, label='SGL', color='blue', alpha=0.7)
axs[0].bar(x + bar_width/2, sgau_recall_amazon, width=bar_width, label='SGAU', color='#00ff00', alpha=0.7)
axs[0].set_title('Amazon-Book', fontsize=14)
axs[0].set_ylabel('Recall@20', fontsize=14)
axs[0].set_xlabel('item groups', fontsize=14)
axs[0].set_xticks(x)
axs[0].set_xticklabels(groups_amazon)
axs[0].legend()
# axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# 绘制 Movielens 数据
axs[1].bar(x - bar_width/2, sgl_recall_movielens, width=bar_width, label='SGL', color='blue', alpha=0.7)
axs[1].bar(x + bar_width/2, sgau_recall_movielens, width=bar_width, label='SGAU', color='#00ff00', alpha=0.7)
axs[1].set_title('ML-20M', fontsize=14)
axs[1].set_ylabel('Recall@20', fontsize=14)
axs[1].set_xlabel('item groups', fontsize=14)
axs[1].set_xticks(x)
axs[1].set_xticklabels(groups_movielens)  # 这里修正了拼写错误
axs[1].legend()
# axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()
plt.savefig('sgau_recall_movielens.png', dpi=600)
plt.show()