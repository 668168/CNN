import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

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

train_data = datasets.ImageFolder(train_datadir, transform=train_transforms)

test_data  = datasets.ImageFolder(test_datadir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=1)
test_loader  = torch.utils.data.DataLoader(test_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=1)

for X, y in test_loader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break