import os
import matplotlib.pyplot as plt
# %matplotlib inline # just for jupyter notebook
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
#pip install torchvision
from torchvision import transforms, models, datasets
#https://pytorch.org/docs/stable/torchvision/index.html
# import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

# import dataset
data_dir = 'flower_data'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

# 定义transforms图像操作.
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomRotation(45),
        transforms.CenterCrop(224),  # 从中心裁剪224*224的图
        transforms.RandomHorizontalFlip(p=0.5),  # 50%的概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 50%的概率垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机改变图像的亮度、对比度、饱和度和色调
        # transforms.Grayscale(num_output_channels=3), # 将图像转换为灰度图
        transforms.RandomGrayscale(p=0.1),  # 10%的概率将图像转换为灰度图
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化图像
    ]),
    "valid": transforms.Compose([
        transforms.Resize(256),  # 重置图像大小
        transforms.CenterCrop(224),  # 从中心裁剪224*224的图片
        transforms.ToTensor(),  # 把图像转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化图像
    ])
}

# 加载数据集
batch_size = 32
# image_datasets 是一个dict, 在 for loop 中, key是x, value 是datasets.ImageFolder对象
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "valid"]}
data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def im_convert(tensor):
    """ 展示数据"""

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    # 在NumPy中，clip 函数用于将数组中的元素限制在一定的范围内。
    # 具体来说，所有小于给定下限的元素会被设置为下限值，所有大于给定上限的元素会被设置为上限值。处于这两个边界之间的元素保持不变。
    image = image.clip(0, 1)

    return image


# 展示数据
dataiter = iter(data_loaders['train'])
## pass ...
inputs, classes = next(dataiter)
## AttributeError: '_SingleProcessDataLoaderIter' object has no attribute 'next'
# inputs, classes = dataiter.next()
## pass ...
# inputs, classes = next(iter(data_loaders['valid']))
print(f"{inputs.shape=}, {classes.shape=}")

fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2
for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
    plt.imshow(im_convert(inputs[idx]))
plt.show()

model_name = 'resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
# 是否用人家训练好的特征来做
feature_extract = True  # True: 只更新最后一层的参数，False: 更新所有参数

# 是否使用GPU 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_on_GPU = torch.cuda.is_available()
print(f"{device=}, {train_on_GPU=}")
if not train_on_GPU:
    print(f"CUDA is not available, training on CPU...")
else:
    print("CUDA is available, training on GPU...")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


model_ft = models.resnet50(pretrained=True)  # 加载预训练模型, 用于迁移学习. pretrained=True 表示加载预训练模型.
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, len(class_names)),
    nn.LogSoftmax(dim=1)
)
model_ft = model_ft.to(device)
input_size = 224

file_name = "checkpoint.pth"

# 是否训练所有层
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  #学习率每7个epoch衰减成原来的1/10
#最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()


# 训练模型


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=file_name):
    since = time.time()
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()  # 反向传播, 计算梯度, 更新权重.
                        optimizer.step()   # 优化器更新参数.

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs
