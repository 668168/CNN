import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from torchvision import datasets
from torchvision.transforms import ToTensor

import math
import numpy as np

# STEP1. initial Parameters. - 设置超参数.
print(f"{'=' * 10} # STEP1. initial Parameters. {'=' * 20}")
loss_func = F.cross_entropy
lr = 0.1  # learning rate
epochs = 5  # how many epochs to train for
bs = 16  # batch size
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


# STEP2. Download training data from open datasets.
######################################
# 1. 先下载dataset ds
# 2. 然后通过get_data , 用DataLoader将ds转换为dl
# 3. 然后通过WrappedDataLoader, 把dl的数据做预处理view的动作.
######################################
print(f"{'=' * 10} # STEP2. Download training data from open datasets. {'=' * 20}")
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
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
print(f"before wrap: imgs.shape={imgs.shape}, labels={labels}")

 # 指定图片大小，图像大小为20宽、5高的绘图(单位为英寸inch)
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

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield self.func(*b)


train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

imgs, labels = next(iter(train_dl))
print(f"after wrap: imgs.shape={imgs.shape}, labels={labels}")


# STEP3. Define model
print(f"{'=' * 10} # STEP3. Define model {'=' * 20}")

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    # Sequential的最后一个元素是AdaptiveAvgPool2d, 通过Lambda做view的动作.
    # Apply a user-defined function as a transform.
    Lambda(lambda x: x.view(x.size(0), -1)),
).to(dev)


# STEP4. Define loss function with LOSS_BATCH
print(f"{'=' * 10} # STEP4. Define loss function with LOSS_BATCH {'=' * 20}")
def loss_batch(model, loss_func, xb, yb, opt=None):
    xb, yb = xb.to(dev), yb.to(dev)
    # cal the loss value
    loss = loss_func(model(xb), yb)

    if opt is not None:
        # cal the gradient
        loss.backward()
        # update the weight
        opt.step()
        # zero the gradient
        opt.zero_grad()

    return loss.item(), len(xb)


# STEP5. Define fit function.
print(f"{'=' * 10} # STEP5. Define fit function. {'=' * 20}")
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f"{epoch+1}/{epochs}, {val_loss=}")


# STEP6. Train the model.
print(f"{'=' * 10} # STEP6. Train the model {epochs} Times. {'=' * 20}")
# opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
opt = optim.Adam(model.parameters(), lr=lr)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
