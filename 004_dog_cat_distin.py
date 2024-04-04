import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.transforms as transforms

import math
import numpy as np

# STEP1. initial Parameters. - 设置**超**参数.
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

train_datadir = './1-cat-dog/train/'
test_datadir  = './1-cat-dog/val/'

train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    # transforms.RandomRotation(degrees=(-10, 10)),  #随机旋转，-10到10度之间随机选
    # transforms.RandomHorizontalFlip(p=0.5),  #随机水平翻转 选择一个概率概率
    # transforms.RandomVerticalFlip(p=0.5),  #随机垂直翻转
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0), # 随机视角
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  #随机选择的高斯模糊模糊图像
    transforms.ToTensor(),          # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    transforms.Normalize(           # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
])

test_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    transforms.ToTensor(),          # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    transforms.Normalize(           # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
])

# datasets.ImageFolder(train_datadir,transform=train_transforms)
# 会根据sub folder 映射对应label, 从0, 开始递增.

train_data = datasets.ImageFolder(train_datadir,transform=train_transforms)

train_labels = train_data.targets
print(f"train_labels={train_labels}")

test_data  = datasets.ImageFolder(test_datadir,transform=test_transforms)

train_dl = torch.utils.data.DataLoader(train_data,
                                          batch_size=bs,
                                          shuffle=True,
                                          num_workers=1)
valid_dl = torch.utils.data.DataLoader(test_data,
                                          batch_size=bs,
                                          shuffle=True,
                                          num_workers=1)

# STEP3. Define model
print(f"{'=' * 10} # STEP3. Define model {'=' * 20}")

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


model = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5),
    # nn.ReLU(),
    nn.Conv2d(6, 16, kernel_size=5),
    # nn.ReLU(),
    # nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    # nn.ReLU(),
    nn.Linear(16*53*53, 120),
    nn.Linear(120, 84),
    nn.AdaptiveAvgPool2d(1),
    # Sequential的最后一个元素是AdaptiveAvgPool2d, 通过Lambda做view的动作.
    Lambda(lambda x: x.view(x.size(0), -1)),
).to(dev)


# STEP4. Define loss function with LOSS_BATCH
print(f"{'=' * 10} # STEP4. Define loss function with LOSS_BATCH {'=' * 20}")
def loss_batch(model, loss_func, xb, yb, opt=None):
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


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(dev), y.to(dev)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# STEP6. Train the model.
print(f"{'=' * 10} # STEP6. Train the model {epochs} Times. {'=' * 20}")
# opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
opt = optim.Adam(model.parameters(), lr=lr)
# fit(epochs, model, loss_func, opt, train_dl, valid_dl)
train(train_dl, model, loss_func, opt)
