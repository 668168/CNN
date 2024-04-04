import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms

from torchinfo import summary

import math
import numpy as np

# STEP1. initial Parameters. - 设置超参数.
print(f"{'=' * 10} # STEP1. initial Parameters. {'=' * 20}")
loss_func = F.cross_entropy
lr = 0.1  # learning rate
epochs = 5  # how many epochs to train for
bs = 32  # batch size
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


# STEP2. Download training data from open datasets.
######################################
# 1. 先下载dataset ds
# 2. 然后通过get_data , 用DataLoader将ds转换为dl
# 3. 然后通过WrappedDataLoader, 把dl的数据做预处理view的动作.
######################################
print(f"{'=' * 10} # STEP2. Download training data from open datasets. {'=' * 20}")
### 导入数据
training_data = datasets.MNIST('data',
                                      train=True,
                                      transform=transforms.ToTensor(), # 将数据类型转化为Tensor
                                      download=True)

test_data  = datasets.MNIST('data',
                                      train=False,
                                      transform=transforms.ToTensor(), # 将数据类型转化为Tensor
                                      download=True)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 1, shuffle=True),
    )


train_dl, valid_dl = get_data(training_data, test_data, bs)
print(f"{len(train_dl)=}")
print(f"{len(valid_dl)=}")

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

#  # 指定图片大小，图像大小为20宽、5高的绘图(单位为英寸inch)
# plt.figure(figsize=(20, 5))
# for i, img in enumerate(imgs[:20]):
#     # 维度缩减
#     npimg = np.squeeze(img.numpy())
#     # 将整个figure分成2行10列，绘制第i+1个子图。
#     plt.subplot(2, 10, i+1)
#     plt.imshow(npimg, cmap=plt.cm.binary)
#     plt.axis('on')
#
# plt.show()
# plt.close(fig)

# STEP3. Define model
print(f"{'=' * 10} # STEP3. Define model {'=' * 20}")

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


model = nn.Sequential(
    # input [32, 1, 28, 28] 32个样本, 1个channel, 28*28的图片
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
    # output [32, 32, 26, 26] 32个样本, 32个channel, 26*26的图片
    nn.ReLU(),
    nn.MaxPool2d(2),
    # output [32, 32, 13, 13] 32个样本, 32个channel, 13*13的图片
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
    # output [32, 64, 11, 11] 32个样本, 64个channel, 11*11的图片
    nn.ReLU(),
    nn.MaxPool2d(2),
    # output [32, 64, 5, 5] 32个样本, 64个channel, 5*5的图片
    # Apply a user-defined function as a transform.
    Lambda(lambda x: x.view(x.size(0), -1)),
    # output [32, 1600] 32个样本, 1600个特征
    nn.Linear(1600, 64),
    # output [32, 64] 32个样本, 64个特征
    nn.ReLU(),
    nn.Linear(64, 10),
    # output [32, 10] 32个样本, 10个特征

).to(dev)

print(f"summary(model): \n{summary(model)}")


# STEP4. Define train function.
def train(train_dl, model, loss_func, opt):
    print(f"{'=' * 10} # STEP4. Define train function. {'=' * 20}")
    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率
    number_bitchs = len(train_dl)
    data_size = len(train_dl.dataset)
    print(f"{data_size=}, {number_bitchs=}, {bs=}")
    for xb, yb in train_dl:
        xb, yb = xb.to(dev), yb.to(dev)
        # cal the loss value
        pred = model(xb)
        loss = loss_func(pred, yb)

        if opt is not None:
            # cal the gradient
            loss.backward()
            # update the weight
            opt.step()
            # zero the gradient
            opt.zero_grad()

            # 记录acc与loss
            train_acc += (pred.argmax(1) == yb).type(torch.float).sum().item()
            train_loss += loss.item()
    train_acc /= data_size
    train_loss /= number_bitchs
    # print(f"in model.train(),  {train_acc=}, {train_loss=}")
    return train_acc, train_loss


# STEP5. Define Validation function.
def test(valid_dl, model, loss_func):
    print(f"{'=' * 10} # STEP5. Define Validation function. {'=' * 20}")
    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率
    number_bitchs = len(valid_dl)
    data_size = len(valid_dl.dataset)
    print(f"{data_size=}, {number_bitchs=}, {bs=}")
    with torch.no_grad():
        for xb, yb in valid_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            # cal the loss value
            pred = model(xb)
            loss = loss_func(pred, yb)

            # 记录acc与loss
            train_acc += (pred.argmax(1) == yb).type(torch.float).sum().item()
            train_loss += loss.item()
        train_acc /= data_size
        train_loss /= number_bitchs
        # print(f"in model.eval(), {train_acc=}, {train_loss=}")
        return train_acc, train_loss


# STEP6. Train the model.
print(f"{'=' * 10} # STEP6. Train the model {epochs} Times. {'=' * 20}")

# 神奇, SGD的准确率到94%, 但Adam 的准确率只有11%.
# SGD 参数加了momentum=0.9, 效果很随机. 移除后正常.可以作为案例面试时候分享.
# 动量（Momentum）的工作原理：
# 动量方法借鉴了物理中的“动量”概念，模拟了一个球在有摩擦的表面上滚动的过程。
# 在这个过程中，球不仅受到当前梯度的影响，也会保留之前速度的一部分，
# 这样可以在某个方向上积累速度，从而加快收敛速度并帮助跳出局部最小值。
# opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# opt = optim.SGD(model.parameters(), lr=lr)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

# opt = optim.Adam(model.parameters(), lr=lr)


epochs = 5
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    model.train()
    epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_func, opt)

    model.eval()
    epoch_test_acc, epoch_test_loss = test(valid_dl, model, loss_func)

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%，Test_loss:{:.3f}')
    print(template.format(epoch + 1, epoch_train_acc * 100, epoch_train_loss, epoch_test_acc * 100, epoch_test_loss))
print('Done')

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