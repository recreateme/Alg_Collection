import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义模型
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super(NCF, self).__init__()
        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP 层
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, user_ids, item_ids):
        # 获取用户和物品的嵌入
        user_vecs = self.user_embedding(user_ids)      # (batch_size, embedding_dim)
        item_vecs = self.item_embedding(item_ids)      # (batch_size, embedding_dim)

        # 拼接用户和物品的嵌入
        x = torch.cat([user_vecs, item_vecs], dim=1)    # (batch_size, 2 * embedding_dim)

        # MLP 前向传播
        x = torch.relu(self.fc1(x))         # (batch_size, hidden_dim)
        x = self.fc2(x)                     # (batch_size, 1)

        return x


# 超参数
num_users = 1000  # 用户数量
num_items = 500  # 物品数量
embedding_dim = 32  # 嵌入维度
hidden_dim = 64  # 隐藏层维度
learning_rate = 0.001
num_epochs = 10

# 创建模型
model = NCF(num_users, num_items, embedding_dim, hidden_dim)
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 假设的训练数据
# user_ids 和 item_ids 是用户和物品的索引，ratings 是对应的评分（0或1）
user_ids = torch.LongTensor(np.random.randint(0, num_users, size=1000))
item_ids = torch.LongTensor(np.random.randint(0, num_items, size=1000))
ratings = torch.FloatTensor(np.random.randint(0, 2, size=1000))

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(user_ids, item_ids).squeeze()  # 预测评分
    loss = criterion(outputs, ratings)

    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

# 使用模型进行预测
model.eval()
with torch.no_grad():
    user_id = torch.LongTensor([0])  # 测试用户
    item_id = torch.LongTensor([0])  # 测试物品
    prediction = model(user_id, item_id)
    print(f'Predicted rating for user {user_id.item()} and item {item_id.item()}: {prediction.item():.4f}')
