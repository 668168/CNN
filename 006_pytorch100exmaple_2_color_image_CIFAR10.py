import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets
from torchvision import transforms

# from torchinfo import summary
from torchsummary import summary

import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"{device=}")

train_ds = torchvision.datasets.CIFAR10('data',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),  # 将数据类型转化为Tensor
                                        download=True)

test_ds = torchvision.datasets.CIFAR10('data',
                                     train=False,
                                     transform=torchvision.transforms.ToTensor(),  # 将数据类型转化为Tensor
                                     download=True)

batch_size = 32

train_dl = torch.utils.data.DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True)

test_dl = torch.utils.data.DataLoader(test_ds,
                                      batch_size=batch_size)

# 取一个批次查看数据格式
# 数据的shape为：[batch_size, channel, height, weight]
# 其中batch_size为自己设定，channel，height和weight分别是图片的通道数，高度和宽度。

# 取一个批次查看数据格式
# 数据的shape为：[batch_size, channel, height, weight]
# 其中batch_size为自己设定，channel，height和weight分别是图片的通道数，高度和宽度。
# 这段代码next(iter(train_dl))用于从数据加载器train_dl中获取第一个批次的数据。
# 这里，train_dl很可能是一个PyTorch的DataLoader实例，用于批量加载训练数据。
# 这段代码的具体作用可以分解为两部分理解：
# iter(train_dl): DataLoader实例是可迭代的，
# iter(train_dl)调用会返回一个迭代器，它按顺序生成数据加载器中的数据批次。
# 每个批次包含了一组样本及其对应的标签，具体组织形式取决于DataLoader的配置。

# next(...): next函数接受一个迭代器作为输入，并返回迭代器的下一个元素。
# 在这个上下文中，它返回数据加载器迭代器的第一个元素，即第一个数据批次。
imgs, labels = next(iter(train_dl))
print(f"imgs.shape={imgs.shape}, labels={labels}")

# 指定图片大小，图像大小为20宽、5高的绘图(单位为英寸inch)
plt.figure(figsize=(20, 5))
for i, imgs in enumerate(imgs[:20]):
    # 维度缩减
    npimg = imgs.numpy().transpose((1, 2, 0))
    # 将整个figure分成2行10列，绘制第i+1个子图。
    plt.subplot(2, 10, i + 1)
    plt.imshow(npimg, cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(labels[i].item())
plt.show()

num_classes = 10  # 图片的类别数


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # H_out = (H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # 不管输入特征图input_tensor的尺寸是多少，
            # 使用AdaptiveAvgPool2d((5, 7))之后，输出特征图的大小将会被调整为5x7。
            # nn.AdaptiveAvgPool2d(1)
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


model = SequentialModel().to(device)
# summary(model)
summary(model, (3, 32, 32), 64)

loss_fn    = nn.CrossEntropyLoss() # 创建损失函数
learn_rate = 1e-2 # 学习率
opt        = torch.optim.SGD(model.parameters(), lr=learn_rate)


# 训练循环
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 训练集的大小，一共60000张图片
    num_batches = len(dataloader)  # 批次数目，1875（60000/32）

    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率

    for X, y in dataloader:  # 获取图片及其标签
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)  # 网络输出
        loss = loss_fn(pred, y)  # 计算网络输出和真实值之间的差距，targets为真实值，计算二者差值即为损失

        # 反向传播
        optimizer.zero_grad()  # grad属性归零
        loss.backward()  # 反向传播
        optimizer.step()  # 每一步自动更新

        # 记录acc与loss
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

    train_acc /= size
    train_loss /= num_batches

    return train_acc, train_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 测试集的大小，一共10000张图片
    num_batches = len(dataloader)  # 批次数目，313（10000/32=312.5，向上取整）
    test_loss, test_acc = 0, 0

    # 当不进行训练时，停止梯度更新，节省计算内存消耗
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)

            # 计算loss
            target_pred = model(imgs)
            loss = loss_fn(target_pred, target)

            test_loss += loss.item()
            test_acc += (target_pred.argmax(1) == target).type(torch.float).sum().item()

    test_acc /= size
    test_loss /= num_batches

    return test_acc, test_loss


epochs = 10
train_loss = []
train_acc = []
test_loss = []
test_acc = []


for epoch in range(epochs):
    model.train()
    epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_fn, opt)

    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_fn)

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%，Test_loss:{:.3f}')
    print(template.format(epoch + 1, epoch_train_acc * 100, epoch_train_loss, epoch_test_acc * 100, epoch_test_loss))
print('Done')


# 绘制训练过程中的损失和正确率
import matplotlib.pyplot as plt
#隐藏警告
import warnings
warnings.filterwarnings("ignore")               #忽略警告信息
plt.rcParams['font.sans-serif']    = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
plt.rcParams['figure.dpi']         = 100        #分辨率

epochs_range = range(epochs)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()