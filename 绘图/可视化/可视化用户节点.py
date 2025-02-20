import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 设置随机种子以便复现
np.random.seed(42)

# 生成20000个用户的64维嵌入
num_users = 2000
embedding_dim = 64
user_embeddings = np.random.rand(num_users, embedding_dim)

# 可选：使用PCA先降低维度
pca = PCA(n_components=50)  # 将维度减少到50
user_embeddings_reduced = pca.fit_transform(user_embeddings)

# 使用t-SNE降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
user_embeddings_2d = tsne.fit_transform(user_embeddings_reduced)

# 绘制二维嵌入
plt.figure(figsize=(10, 10))
plt.scatter(user_embeddings_2d[:, 0], user_embeddings_2d[:, 1], alpha=0.5)
plt.title('t-SNE Visualization of User Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid()
plt.show()