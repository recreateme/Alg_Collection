import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 假设你已经有了用户或物品的嵌入
# 这里用随机数据作为示例
num_samples = 100  # 嵌入的样本数量
embedding_dim = 10  # 嵌入的维度
embeddings = np.random.rand(num_samples, embedding_dim)

# 使用 t-SNE 降维
for per in range(5, 100, 5):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
plt.title('t-SNE Visualization of LightGCN Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid()
plt.show()
