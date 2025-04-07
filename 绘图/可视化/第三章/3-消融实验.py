import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['KACR-inter', 'KACR-ssl', 'KACR']
metrics = ['Recall@20', 'NDCG@20']  # 修改为 Recall@20 和 NDCG@20

# Amazon-Book 数据
recall_amazon = [0.1618, 0.1527, 0.1815]  # Amazon-Book 的 Recall
ndcg_amazon = [0.1086, 0.1029, 0.1189]  # Amazon-Book 的 NDCG

# LastFM 数据
recall_lastfm = [0.0932, 0.0868, 0.1056]  # LastFM 的 Recall
ndcg_lastfm = [0.1366, 0.1310, 0.1490]   # LastFM 的 NDCG

# 设置柱状图的宽度和位置
bar_width = 0.2  # 减小柱状图宽度
x = np.arange(len(metrics))  # 横坐标为指标（Recall@20, NDCG@20）

# 定义填充样式（hatch）来区分模型
hatch_styles = ['//', '\\\\', '||']  # 分别为 KACR-inter、KACR-ssl、KACR

# 定义颜色列表
# colors = ['blue', 'green', 'red']  # 分别为 KACR-inter、KACR-ssl、KACR
# 定义颜色列表（柔和颜色）
colors = ['#add8e6', '#90ee90', '#ffe4c4']  # 分别为 KACR-inter、KACR-ssl、KACR

# 创建两个子图（一个用于 Amazon-Book，一个用于 LastFM）
# 增大图表高度以美观
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

fontsize = 14  # 基础字体大小

# 绘制 Amazon-Book 图表
for i, model in enumerate(models):
    ax1.bar(x[0] - bar_width + i * bar_width, recall_amazon[i] * 100, bar_width,
            label=model, hatch=hatch_styles[i], edgecolor='black', color=colors[i], alpha=0.8)
    ax1.bar(x[1] - bar_width + i * bar_width, ndcg_amazon[i] * 100, bar_width,
            hatch=hatch_styles[i], edgecolor='black', color=colors[i], alpha=0.8)

# 绘制 LastFM 图表
for i, model in enumerate(models):
    ax2.bar(x[0] - bar_width + i * bar_width, recall_lastfm[i] * 100, bar_width,
            label=model, hatch=hatch_styles[i], edgecolor='black', color=colors[i], alpha=0.8)
    ax2.bar(x[1] - bar_width + i * bar_width, ndcg_lastfm[i] * 100, bar_width,
            hatch=hatch_styles[i], edgecolor='black', color=colors[i], alpha=0.8)

# 设置轴标签和图例的字体大小
for i,ax in enumerate([ax1, ax2]):
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=fontsize)
    if i%2==0:
        ax.legend(fontsize=fontsize, loc='upper right', edgecolor='black')  # 图例边框为黑色
    else:
        ax.legend(fontsize=fontsize, loc='upper left', edgecolor='black')
    # 去掉左右纵轴的刻度和标签
    ax.set_yticklabels([])  # 去掉左侧纵坐标刻度标签
    ax_n = ax.twinx()
    ax_n.set_yticklabels([])  # 去掉右侧纵坐标刻度标签

    # 增大纵轴显示范围（乘以 1.5 倍）
    ax.set_ylim(0, max([max(recall_amazon), max(ndcg_amazon), max(recall_lastfm), max(ndcg_lastfm)]) * 100 * 1.15)
    ax_n.set_ylim(0, max([max(recall_amazon), max(ndcg_amazon), max(recall_lastfm), max(ndcg_lastfm)]) * 100 * 1.15)

# 添加数值标签，并设置字体大小，乘以100后显示数值
for i, ax in enumerate([ax1, ax2]):
    for j, metric in enumerate(metrics):
        for k, model in enumerate(models):
            if i == 0:  # Amazon-Book
                value = recall_amazon[k] if metric == 'Recall@20' else ndcg_amazon[k]
            else:  # LastFM
                value = recall_lastfm[k] if metric == 'Recall@20' else ndcg_lastfm[k]
            # 调整文字位置，确保在图表内部
            ax.text(j - bar_width + k * bar_width, value * 100 + 1.0,  # 增大偏移量
                    f'{value * 100:.2f}', ha='center', va='bottom', fontsize=fontsize,
                    bbox=None)  # 去掉数值标签的框

# 为两个子图添加横轴的数据集标签
ax1.set_title('Amazon-Book', fontsize=fontsize, pad=20)  # 添加数据集标签
ax2.set_title('Last-FM', fontsize=fontsize, pad=20)  # 添加数据集标签

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()