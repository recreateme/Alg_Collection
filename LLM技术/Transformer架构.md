以下是基于多个来源整合的Transformer架构完整实现代码，包含编码器、解码器、多头注意力等核心模块，并附带关键实现说明：

---

一、核心组件实现
1. 多头注意力机制（Multi-Head Attention）
```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制模块"""
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        """
        初始化参数:
        :param d_model: 模型总维度（需能被num_heads整除）
        :param num_heads: 注意力头的数量
        :param dropout: 注意力权重丢弃率
        """
        super().__init__()
        assert d_model % num_heads == 0, "模型维度必须能被头的数量整除"
        
        # 基础参数设置
        self.d_k = d_model // num_heads    # 每个注意力头的维度
        self.num_heads = num_heads         # 多头数量
        
        # 定义线性变换层（Q/K/V的生成矩阵）
        self.wq = nn.Linear(d_model, d_model)  # 查询向量变换
        self.wk = nn.Linear(d_model, d_model)  # 键向量变换
        self.wv = nn.Linear(d_model, d_model)  # 值向量变换
        
        # 正则化组件
        self.dropout = nn.Dropout(dropout)     # 注意力权重随机丢弃
        self.fc = nn.Linear(d_model, d_model) # 最终输出线性变换

    def forward(self, query, key, value, mask=None):
        """
        前向传播过程:
        :param query: 查询向量 [batch_size, seq_len, d_model]
        :param key: 键向量 [batch_size, seq_len, d_model]
        :param value: 值向量 [batch_size, seq_len, d_model]
        :param mask: 掩码张量（可选）[batch_size, 1, seq_len, seq_len]
        :return: 注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)  # 获取批量大小
        
        # ========== 多头拆分 ==========
        # 线性变换 + 维度重塑 [batch_size, num_heads, seq_len, d_k]
        Q = self.wq(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.wk(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.wv(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        
        # ========== 注意力计算 ==========
        # 计算缩放点积得分 [batch_size, num_heads, seq_len, seq_len],获取不同注意力头的序列间系数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（使无效位置的注意力权重趋近于0）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 用极小值填充掩码位置
            
        # 归一化得到注意力权重
        attn = torch.softmax(scores, dim=-1)  # 最后一个维度做softmax
        attn = self.dropout(attn)             # 随机丢弃部分注意力权重
        
        # ========== 上下文计算 ==========
        # 注意力权重与Value相乘 [batch_size, num_heads, seq_len, d_k]
        context = torch.matmul(attn, V)
        
        # ========== 多头合并 ==========
        # 维度调整 [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous()  # 保证内存连续
        context = context.view(batch_size, -1, self.num_heads * self.d_k)
        
        # 最终线性变换
        output = self.fc(context)
        return output
```

2. 位置编码（Positional Encoding）
```python
import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 初始化位置编码矩阵，形状为[max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引，形状为[max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母项，形状为[d_model//2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # 为位置编码矩阵的偶数列赋值正弦值，形状为[max_len, d_model//2]
        pe[:, 0::2] = torch.sin(position * div_term)
        # 为位置编码矩阵的奇数列赋值余弦值，形状为[max_len, d_model//2]
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将位置编码矩阵添加到buffer中，形状为[1, max_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x的形状为[batch_size, seq_len, d_model]
        # pe的形状为[1, max_len, d_model]，取前seq_len个位置，形状为[1, seq_len, d_model]
        # 将位置编码加到输入x上，返回形状为[batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]
```

---

二、编码器与解码器层
1. 编码器层（Encoder Layer）
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        """
        Transformer编码器层实现
        :param d_model: 模型特征维度（输入/输出维度）
        :param num_heads: 多头注意力的头数
        :param d_ff: 前馈网络隐藏层维度（默认2048）
        :param dropout: 随机失活率（默认0.1）
        """
        super().__init__()
        # 多头自注意力机制模块
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈神经网络模块
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 扩展维度到d_ff
            nn.ReLU(),                  # 非线性激活
            nn.Linear(d_ff, d_model)    # 降维回d_model
        )
        
        # 层归一化模块（Post-LN结构，残差连接后进行归一化）
        self.norm1 = nn.LayerNorm(d_model)  # 注意力层后的归一化
        self.norm2 = nn.LayerNorm(d_model)  # 前馈层后的归一化
        
        # 随机失活层（防止过拟合）
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播过程（实现残差连接和层归一化）
        :param x: 输入张量 [batch_size, seq_len, d_model]
        :param mask: 注意力掩码（可选）[batch_size, 1, seq_len, seq_len]
        :return: 编码后的张量 [batch_size, seq_len, d_model]
        """
        # ========== 自注意力子层 ==========
        # 1. 计算多头注意力（Q=K=V=x）
        attn_output = self.attn(x, x, x, mask)
        
        # 2. 残差连接 + Dropout + 层归一化
        # 原始输入x与注意力输出相加，缓解梯度消失
        x = x + self.dropout(attn_output)  
        # 层归一化保持特征分布稳定
        x = self.norm1(x)                  
        
        # ========== 前馈神经网络子层 ==========
        # 3. 前馈网络计算
        ffn_output = self.ffn(x)
        
        # 4. 残差连接 + Dropout + 层归一化
        # 注意此处残差连接的是前一步归一化后的结果
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x
```

2. 解码器层（Decoder Layer）
```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 自注意力（含未来词掩码） 
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力（编码器-解码器） 
        attn_output = self.cross_attn(enc_output, enc_output, x, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络 
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x
```

---

三、完整Transformer模型
```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 构建编码器堆叠 
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # 构建解码器堆叠 
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask):
        src = self.encoder_embed(src)
        src = self.pos_encoding(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        return src

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        tgt = self.decoder_embed(tgt)
        tgt = self.pos_encoding(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.fc(dec_output)
```

---

四、辅助函数与训练示例
1. 掩码生成函数
```python
def generate_mask(src, tgt, pad_idx=0):
    # 输入序列填充掩码 
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 未来词掩码（下三角矩阵） 
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    seq_len = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask
```

2. 模型初始化与训练
```python
# 超参数设置 
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
dropout = 0.1

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 示例训练循环 
for epoch in range(100):
    src = torch.randint(0, src_vocab_size, (32, 50))  # batch_size=32, seq_len=50
    tgt = torch.randint(0, tgt_vocab_size, (32, 60))
    src_mask, tgt_mask = generate_mask(src, tgt)
    
    outputs = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
    loss = nn.CrossEntropyLoss()(outputs.view(-1, tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

关键实现要点说明
1. 模块化设计：通过分离注意力、位置编码等组件提升代码复用性 
2. 残差连接：每个子层输出与输入相加后接层归一化，防止梯度消失 
3. 掩码机制：包含填充掩码（处理变长序列）和未来词掩码（防止信息泄露）
4. 多头注意力：通过拆分维度实现并行计算，提升模型表达能力 
5. 位置编码：采用正弦/余弦函数生成绝对位置信息 

> 完整实现需配合数据预处理和超参调优，建议参考论文《Attention Is All You Need》进行扩展。代码兼容PyTorch 2.0+环境，可通过调整num_layers等参数适配不同规模任务。