import torch
from torch import nn
from transformers import AutoConfig
from transformers import BertPreTrainedModel, BertModel
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

checkpoint = "bert-base-chinese"


class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]        # 取所有句子的第一个token的向量作为CLS向量
        cls_vectors = self.dropout(cls_vectors)      # dropout正则化
        logits = self.classifier(cls_vectors)       # 2分类
        return logits


config = AutoConfig.from_pretrained(checkpoint)  # 加载配置文件
model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)         # 加载模型，并加载到device上
print(model)          # 打印模型结构


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    '''

    :param dataloader:  数据加载器
    :param model:       模型
    :param loss_fn:     损失函数
    :param optimizer:   优化器
    :param lr_scheduler:   学习率调度器
    :param epoch:       训练epoch
    :param total_loss:   训练损失
    :return:
    '''
    progress_bar = tqdm(range(len(dataloader)))         # 创建进度条
    progress_bar.set_description(f'loss: {0:>7f}')      # 初始化进度条描述信息
    finish_step_num = (epoch - 1) * len(dataloader)     # 计算已经完成的step数

    model.train()      # 开启训练模式
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)        # 训练数据加载到device上
        pred = model(X)                          # 预测
        loss = loss_fn(pred, y)                  # 计算损失

        optimizer.zero_grad()                    # 清空梯度
        loss.backward()                          # 反向传播获取梯度
        optimizer.step()                         # 更新模型参数
        lr_scheduler.step()                      # 更新学习率

        total_loss += loss.item()                # 累计训练损失
        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')  # 更新进度条描述信息
        progress_bar.update(1)                   # 更新进度条
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']     # 确保模型验证集或测试集
    size = len(dataloader.dataset)      # 获取数据集大小
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100 * correct):>0.1f}%\n")
