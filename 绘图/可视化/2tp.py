import pandas as pd
import matplotlib.pyplot as plt

# 假设数据文件已经存在并具有正确的格式
data = pd.read_csv('model_performance.txt', sep='\t')

# 创建画布
fig, axs = plt.subplots(1, 3, figsize=(24, 8))  # 增加宽度以适应更多的细节

# 绘制每个数据集的图表
datasets = ['AmazonBook', 'ML-1M']

for i, dataset in enumerate(datasets):
    # 获取当前子图的主轴
    ax1 = axs[i]

    # 计算 recall 和 ndcg 的最小值和最大值
    recall_min_value = data[f'recall_{dataset}'].min() * 100
    recall_max_value = data[f'recall_{dataset}'].max() * 100
    ndcg_min_value = data[f'ndcg_{dataset}'].min() * 100
    ndcg_max_value = data[f'ndcg_{dataset}'].max() * 100

    # 设置 recall 的 y 轴范围
    recall_y_min = max(0, recall_min_value - (recall_max_value - recall_min_value) * 0.1)
    recall_y_max = min(100, recall_max_value + (recall_max_value - recall_min_value) * 0.1)

    # 设置 ndcg 的 y 轴范围
    ndcg_y_min = max(0, ndcg_min_value - (ndcg_max_value - ndcg_min_value) * 0.1)
    ndcg_y_max = min(100, ndcg_max_value + (ndcg_max_value - ndcg_min_value) * 0.1)

    # 设置 recall 的 y 轴范围
    ax1.set_ylim(recall_y_min, recall_y_max)

    # 使用蓝色绘制 recall 数据
    recall_line, = ax1.plot(data['weight'], data[f'recall_{dataset}'] * 100, label='Recall@20', color='blue', marker='o',
                            linewidth=2)

    # 创建共享 x 轴的第二个 y 轴
    ax2 = ax1.twinx()

    # 设置 ndcg 的 y 轴范围
    ax2.set_ylim(ndcg_y_min, ndcg_y_max)

    # 使用橙色绘制 ndcg 数据
    ndcg_line, = ax2.plot(data['weight'], data[f'ndcg_{dataset}'] * 100, label='NDCG@20', color='orange', marker='x',
                          linewidth=2)

    # 设置坐标轴标签
    ax1.set_xlabel('kge_loss weight')
    ax1.set_ylabel('Recall (%)', color='blue')
    ax2.set_ylabel('NDCG (%)', color='orange')

    # 设置标题
    ax1.set_title(f'{dataset}')

    # 添加图例
    ax1.legend(handles=[recall_line], loc='upper left')
    ax2.legend(handles=[ndcg_line], loc='upper right')

    # 设置 y 轴为百分数格式，并调整刻度
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))

    # 减少刻度数量至大约5个
    ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    ax1.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()
# plt.savefig('model_performance.png', dpi=600)  # 提高保存的分辨率
plt.show()