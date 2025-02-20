import matplotlib.pyplot as plt

# 模型名称
models = ['KGAT', 'KGIN', 'KGCL', 'MCCLK', 'L2CL', 'KA-Rec']

# 召回率和NDCG值
recall_rates = [0.1501, 0.1657, 0.1665, 0.1690, 0.1496, 0.1728]
ndcg_rates = [0.1003, 0.1156, 0.1149, 0.1173, 0.0998, 0.1201]
plt.rcParams['font.family'] = 'SimHei'
# 创建图形和坐标轴
fig, ax = plt.subplots()

# 设置x轴范围
index = range(len(models))

# 绘制召回率折线
bar1 = ax.plot(index, recall_rates, label='Recall', marker='o')

# 绘制NDCG折线
bar2 = ax.plot(index, ndcg_rates, label='NDCG', marker='x')

# 添加模型名称作为x轴标签
plt.xticks(index, models)

# 添加标题和标签
ax.set_title('Model Performance Comparison on Amazon-Book')
ax.set_xlabel('Models')
ax.set_ylabel('Rates')

# 显示图例
ax.legend()

# 显示网格
ax.grid(True)

# 展示图表
plt.tight_layout()
plt.show()