import torch
import torch.nn as nn
from monai.networks.nets import VNet
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord
from monai.losses import DiceLoss
from monai.inferers import SimpleInferer

# 1. 数据准备
# 假设我们有肝脏 CT 数据和对应的分割掩码（NIfTI格式）
data = [
    {"image": "path/to/liver_ct.nii.gz", "label": "path/to/liver_mask.nii.gz"},
    # 添加更多数据样本
]

# 定义数据预处理流程
transforms = Compose([
    LoadImaged(keys=["image", "label"]),  # 加载 NIfTI 文件
    AddChanneld(keys=["image", "label"]),  # 增加通道维度 (B, C, H, W, D)
    ScaleIntensityd(keys=["image"]),  # 归一化图像强度到 [0, 1]
    ToTensord(keys=["image", "label"])  # 转为 PyTorch 张量
])

# 创建数据集和加载器
dataset = Dataset(data=data, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



# 2. 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VNet(
    spatial_dims=3,  # 3D 数据
    in_channels=1,  # 输入通道数（灰度图像）
    out_channels=2,  # 输出通道数（背景+肝脏）
    act="relu",  # 激活函数
    dropout_prob=0.1  # Dropout 防止过拟合
).to(device)

# 3. 定义损失函数和优化器
loss_fn = DiceLoss(to_onehot_y=True, softmax=True)  # Dice Loss，适用于分割
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        images = batch["image"].to(device)  # [B, C, H, W, D]
        labels = batch["label"].to(device)  # [B, C, H, W, D]

        # 前向传播
        outputs = model(images)  # 输出 [B, 2, H, W, D]
        loss = loss_fn(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 5. 推理示例
model.eval()
inferer = SimpleInferer()
with torch.no_grad():
    for batch in dataloader:
        image = batch["image"].to(device)
        pred = inferer(image, model)  # [B, 2, H, W, D]
        pred_mask = torch.argmax(pred, dim=1)  # 转为掩码 [B, H, W, D]
        # 可保存 pred_mask 或可视化

# 6. 可视化（可选）
# 使用 matplotlib 或 nibabel 保存结果为 NIfTI 文件