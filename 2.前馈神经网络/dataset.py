import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import cross_entropy, binary_cross_entropy
from torch.nn import CrossEntropyLoss
from torchvision import transforms
# 数据集定义
# 构建回归数据集合 - traindataloader1, testdataloader1
data_num, train_num, test_num = 10000, 7000, 3000 # 分别为样本总数量，训练集样本数量和测试集样本数量
true_w, true_b = 0.0056 * torch.ones(500,1), 0.028
features = torch.randn(10000, 500)
labels = torch.matmul(features,true_w) + true_b # 按高斯分布
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)
# 划分训练集和测试集
train_features, test_features = features[:train_num,:], features[train_num:,:]
train_labels, test_labels = labels[:train_num], labels[train_num:]
batch_size = 128
traindataset1 = torch.utils.data.TensorDataset(train_features,train_labels)
testdataset1 = torch.utils.data.TensorDataset(test_features, test_labels)
traindataloader1 = torch.utils.data.DataLoader(dataset=traindataset1,batch_size=batch_size,shuffle=True)
testdataloader1 = torch.utils.data.DataLoader(dataset=testdataset1,batch_size=batch_size,shuffle=True)

# 构二分类数据集合
data_num, train_num, test_num = 10000, 7000, 3000  # 分别为样本总数量，训练集样本数量和测试集样本数量
# 第一个数据集 符合均值为 2 标准差为1 得分布
features1 = torch.normal(mean=2, std=1, size=(data_num, 200), dtype=torch.float32)
labels1 = torch.ones(data_num)
# 第二个数据集 符合均值为 -2 标准差为1的分布
features2 = torch.normal(mean=-2, std=1, size=(data_num, 200), dtype=torch.float32)
labels2 = torch.zeros(data_num)

# 构建训练数据集
train_features2 = torch.cat((features1[:train_num], features2[:train_num]), dim=0)  # size torch.Size([14000, 200])
train_labels2 = torch.cat((labels1[:train_num], labels2[:train_num]), dim=-1)  # size  torch.Size([6000, 200])
# 构建测试数据集
test_features2 = torch.cat((features1[train_num:], features2[train_num:]), dim=0)  # torch.Size([14000])
test_labels2 = torch.cat((labels1[train_num:], labels2[train_num:]), dim=-1)  # torch.Size([6000])
batch_size = 128
# Build the training and testing dataset
traindataset2 = torch.utils.data.TensorDataset(train_features2, train_labels2)
testdataset2 = torch.utils.data.TensorDataset(test_features2, test_labels2)
traindataloader2 = torch.utils.data.DataLoader(dataset=traindataset2,batch_size=batch_size,shuffle=True)
testdataloader2 = torch.utils.data.DataLoader(dataset=testdataset2,batch_size=batch_size,shuffle=True)

# 定义多分类数据集 - train_dataloader - test_dataloader
batch_size = 128
# Build the training and testing dataset
traindataset3 = torchvision.datasets.FashionMNIST(root='E:\\DataSet\\FashionMNIST\\Train',
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.ToTensor())
testdataset3 = torchvision.datasets.FashionMNIST(root='E:\\DataSet\\FashionMNIST\\Test',
                                                 train=False,
                                                 download=True,
                                                 transform=transforms.ToTensor())
traindataloader3 = torch.utils.data.DataLoader(traindataset3, batch_size=batch_size, shuffle=True)
testdataloader3 = torch.utils.data.DataLoader(testdataset3, batch_size=batch_size, shuffle=False)
# 绘制图像的代码
def picture(num, trainl, testl):
    plt.title('Figure ' + str(num))
    plt.plot(trainl, c='g', label='Train Loss')
    plt.plot(testl, c='r', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
print(f'回归数据集   样本总数量{len(traindataset1) + len(testdataset1)},训练样本数量{len(traindataset1)},测试样本数量{len(testdataset1)}')
print(f'二分类数据集 样本总数量{len(traindataset2) + len(testdataset2)},训练样本数量{len(traindataset2)},测试样本数量{len(testdataset2)}')
print(f'多分类数据集 样本总数量{len(traindataset3) + len(testdataset3)},训练样本数量{len(traindataset3)},测试样本数量{len(testdataset3)}')