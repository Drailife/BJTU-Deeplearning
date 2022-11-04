import matplotlib.pyplot as plt

from dataset import *
# import torch.nn as nn
# 利用torch.nn实现前馈神经网络-多分类任务
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
# 定义自己的前馈神经网络
class MyNet23(nn.Module):
    def __init__(self):
        super(MyNet23, self).__init__()
        # 设置隐藏层和输出层的节点数
        num_inputs, num_hiddens, num_outputs = 28*28, 256, 10
        # 定义模型结构
        self.input_layer = nn.Flatten()
        self.hidden_layer = nn.Linear(num_inputs, num_hiddens)
        self.output_layer = nn.Linear(num_hiddens, num_outputs)
        self.relu = nn.ReLU()

    def logistic(self, x):  # 定义logistic函数
        x = 1.0 / (1.0 + torch.exp(-x))
        return x
    # 定义前向传播
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(self.hidden_layer(x))
        x = self.logistic(self.output_layer(x))
        return x

# 训练
model23 = MyNet23()  # logistics模型
optimizer = SGD(model23.parameters(), lr=0.05)  # 优化函数
epochs = 100  # 训练轮数
criterion = CrossEntropyLoss()
train_all_loss23 = []  # 记录训练集上得loss变化
test_all_loss23 = []  # 记录测试集上的loss变化
begintime23 = time.time()
for epoch in range(epochs):
    train_l = 0
    for data, labels in traindataloader3:
        pred = model23(data)
        train_each_loss = criterion(pred.view(-1), labels.view(-1))  # 计算每次的损失值
        optimizer.zero_grad()  # 梯度清零
        train_each_loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新
        train_l += train_each_loss.item()
    train_all_loss23.append(train_l)  # 添加损失值到列表中
    with torch.no_grad():
        test_loss = 0
        for data, labels in testdataloader3:
            pred = model23(data)
            test_each_loss = criterion(pred.view(-1),labels)
            test_loss += test_each_loss.item()
        test_all_loss23.append(test_loss)
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print('epoch: %d | train loss:%.5f | test loss:%.5f' % (epoch + 1, train_all_loss23[-1], test_all_loss23[-1]))
endtime23 = time.time()
print("torch.nn实现前馈网络-回归实验 %d轮 总用时: %.3fs" % (epochs, endtime23 - begintime23))
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(train_all_loss23)
plt.subplot(122)
plt.plot(test_all_loss23)
plt.show()