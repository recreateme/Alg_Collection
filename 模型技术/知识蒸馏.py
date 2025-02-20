import torch
import torch.nn as nn
import torch.optim as optim


# 定义教师模型和学生模型
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


# 初始化模型
teacher = TeacherModel()
student = StudentModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student.parameters(), lr=0.001)

# 假设有一些输入数据和标签
inputs = torch.randn(32, 10)  # 32个样本，10个特征
true_labels = torch.randint(0, 5, (32,))  # 真实标签

# 获取教师模型的输出
with torch.no_grad():
    teacher_outputs = teacher(inputs)

# 学生模型的训练
for epoch in range(100):  # 训练100个epoch
    optimizer.zero_grad()

    # 学生模型的输出
    student_outputs = student(inputs)

    # 计算损失
    loss_student = criterion(student_outputs, true_labels)        # 学生模型的损失
    loss_teacher = criterion(student_outputs, torch.softmax(teacher_outputs / 2, dim=1).argmax(dim=1))          # 教师模型的损失

    # 总损失
    alpha = 0.5
    loss = alpha * loss_student + (1 - alpha) * loss_teacher

    # 反向传播
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
