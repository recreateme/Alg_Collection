import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class WideAndDeepModel(nn.Module):
    def __init__(self,
                 wide_features,  # 宽度特征的数量
                 deep_features,  # 深度特征的数量
                 embed_dims,  # 字典格式，key是类别特征名，value是embedding维度
                 hidden_units=[256, 128, 64],  # Deep部分的隐藏层单元数
                 dropout=0.3):  # Dropout比率
        super(WideAndDeepModel, self).__init__()

        # Wide部分
        self.wide = nn.Linear(wide_features, 1)

        # Deep部分
        # 1. Embedding层
        self.embeddings = nn.ModuleDict()
        total_embed_dims = 0
        for feat, dim in embed_dims.items():
            self.embeddings[feat] = nn.Embedding(deep_features, dim)
            total_embed_dims += dim

        # 2. 全连接层
        self.linears = nn.ModuleList()
        prev_dim = total_embed_dims

        for hidden_unit in hidden_units:
            self.linears.append(nn.Linear(prev_dim, hidden_unit))
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Dropout(dropout))
            prev_dim = hidden_unit

        self.deep_output = nn.Linear(hidden_units[-1], 1)

        # 最终输出层
        self.final_activation = nn.Sigmoid()

    def forward(self, wide_input, deep_categorical_inputs):
        # Wide路径
        wide_output = self.wide(wide_input)

        # Deep路径
        # 1. 处理embedding
        embed_outputs = []
        for feat, embed_layer in self.embeddings.items():
            embed_output = embed_layer(deep_categorical_inputs[feat])
            embed_outputs.append(embed_output)

        deep_input = torch.cat(embed_outputs, dim=1)

        # 2. 通过深度网络
        deep_output = deep_input
        for layer in self.linears:
            deep_output = layer(deep_output)

        deep_output = self.deep_output(deep_output)

        # 组合wide和deep的输出
        combined_output = wide_output + deep_output
        return self.final_activation(combined_output)


# 训练函数
def train_wide_and_deep(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (wide_features, deep_features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(wide_features, deep_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')


# 示例使用代码
def example_usage():
    # 假设数据
    wide_features = 10  # wide特征数量
    deep_features = 100  # 每个类别特征的可能取值数量
    embed_dims = {
        'category1': 8,
        'category2': 8,
        'category3': 8
    }

    # 初始化模型
    model = WideAndDeepModel(
        wide_features=wide_features,
        deep_features=deep_features,
        embed_dims=embed_dims
    )

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 假设有训练数据加载器
    # train_loader = ...

    # 训练模型
    # train_wide_and_deep(model, train_loader, optimizer, criterion)