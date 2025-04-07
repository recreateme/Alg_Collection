import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import faiss

faiss.normalize_L2()

import accelerate
accelerator = accelerate.Accelerator()
accelerator.pad_across_processes()

class PositionalEncoding(nn.Module):
    # 为了给模型提供序列中每个词的位置信息，我们需要添加位置编码
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 注册为持久缓存，不参与梯度计算
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return x


class MultiHeadAttention(nn.Module):
    # Transformer使用多头注意力机制来捕捉不同子空间中的信息
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Tensor, shape [batch_size, seq_len, d_model]
            key: Tensor, shape [batch_size, seq_len, d_model]
            value: Tensor, shape [batch_size, seq_len, d_model]
            mask: Tensor or None, shape [batch_size, seq_len, seq_len]
        Returns:
            output: Tensor, shape [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        # 线性变换并分割成多头
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out_linear(attn_output)
        return output


class FeedForward(nn.Module):
    # 前馈神经网络，Transformer层包含一个前馈神经网络，通常由两个线性层和一个ReLU激活函数组成
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            output: Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, d_model]
            src_mask: Tensor or None, shape [batch_size, seq_len, seq_len]
        Returns:
            output: Tensor, shape [batch_size, seq_len, d_model]
        """
        # 自注意力机制
        attn_output = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        # 前馈神经网络
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src



class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        """
        Args:
            tgt: Tensor, shape [batch_size, tgt_seq_len, d_model]
            memory: Tensor, shape [batch_size, src_seq_len, d_model]
            tgt_mask: Tensor or None, shape [batch_size, tgt_seq_len, tgt_seq_len]
            memory_mask: Tensor or None, shape [batch_size, tgt_seq_len, src_seq_len]
        Returns:
            output: Tensor, shape [batch_size, tgt_seq_len, d_model]
        """
        # 自注意力机制
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)
        # 交叉注意力机制
        attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)
        # 前馈神经网络
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, vocab_size, max_len,
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        """生成掩码矩阵"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        """
        Args:
            src: Tensor, shape [batch_size, src_seq_len]
            tgt: Tensor, shape [batch_size, tgt_seq_len]
        Returns:
            output: Tensor, shape [batch_size, tgt_seq_len, vocab_size]
        """
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # 嵌入和位置编码
        src = self.dropout(self.positional_encoding(self.embedding(src)))
        tgt = self.dropout(self.positional_encoding(self.embedding(tgt)))

        # 编码器
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # 解码器
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        # 输出层
        output = self.fc_out(tgt)
        return output
