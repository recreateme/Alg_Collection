import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保可重复性
np.random.seed(42)

# 模拟数据
epochs = np.arange(1, 101)

# 定义每个模型的初始损失和收敛值
initial_losses = [0.82, 0.88, 0.83, 0.96, 0.93]
convergence_values = [0.12, 0.15, 0.13, 0.11, 0.14]  # 每个模型的收敛值
losses = []

for i, initial_loss in enumerate(initial_losses):
    # 根据收敛值调整损失函数
    decay_rate = 0.03 + (0.01 * i)  # 不同的衰减速率
    loss = initial_loss * np.exp(-decay_rate * epochs) + convergence_values[i] * (1 - np.exp(-0.02 * epochs))

    noise_magnitude = 0.002 * np.exp((100 - epochs) / 100)  # 随着epoch增加，噪声减少
    loss += np.random.normal(0, noise_magnitude, len(epochs))
    losses.append(np.clip(loss, 0.01, None))  # 确保损失不低于0.005

# 绘制损失曲线
plt.figure()
colors = ['blue', 'orange', 'green', 'red', 'purple']

model = ["KGAT", "RippleNet", "CKE", "KGIN", "KGCF"]
for i, loss in enumerate(losses):
    plt.plot(epochs, loss, label=f'{model[i]}', color=colors[i])

# 添加图表元素
plt.title('Amazon-Book CF Loss')
plt.xlabel('Epochs')
plt.ylabel('CF Loss')
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('loss_decrease.png')
plt.show()
