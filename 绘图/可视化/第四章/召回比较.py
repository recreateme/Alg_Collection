import matplotlib.pyplot as plt

# 设置横轴 K 值
K_values = [5, 10, 20, 50, 100]
x_indices = list(range(len(K_values)))  # 等间距索引 [0, 1, 2, 3, 4]

# 为五个模型手动设置 Recall 值（0到1之间）
model1_recall = [0.023, 0.35, 0.5, 0.65, 0.8]   # 模型 1
model2_recall = [0.025, 0.3, 0.45, 0.6, 0.75]  # 模型 2
model3_recall = [0.034, 0.4, 0.55, 0.7, 0.85]  # 模型 3
model4_recall = [0.041, 0.25, 0.4, 0.55, 0.7]   # 模型 4
model5_recall = [0.5, 0.45, 0.6, 0.75, 0.9]   # 模型 5

# 创建折线图
plt.figure(figsize=(6, 4))  # 设置图形大小

# 绘制五条折线，使用等间距的 x_indices
plt.plot(x_indices, model1_recall, label='Model 1', color='blue', linestyle='-', marker='o', linewidth=2)
plt.plot(x_indices, model2_recall, label='Model 2', color='red', linestyle='--', marker='s', linewidth=2)
plt.plot(x_indices, model3_recall, label='Model 3', color='green', linestyle='-.', marker='^', linewidth=2)
plt.plot(x_indices, model4_recall, label='Model 4', color='purple', linestyle=':', marker='d', linewidth=2)
plt.plot(x_indices, model5_recall, label='Model 5', color='orange', linestyle='-', marker='*', linewidth=2)

# 设置横轴和纵轴标签
plt.xlabel('K')
plt.ylabel('Recall')

# 设置横轴刻度为等间距，并替换为 K_values
plt.xticks(x_indices, K_values)

# 添加图例
plt.legend()

# 添加水平网格线
plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.show()