import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torchvision import transforms, datasets

import os,PIL,pathlib

# step1. 基本准备

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# step2. 下载数据集
import os, PIL, random, pathlib

data_dir = './weather/'
data_dir = pathlib.Path(data_dir)

data_paths = list(data_dir.glob('*'))
classeNames = [str(path).split("\\")[1] for path in data_paths]
print(f"{classeNames=}")

# step3. 预览图像
import os
import matplotlib.pyplot as plt
from PIL import Image

# 指定图像文件夹路径
image_folder = './weather/cloudy/'

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

# 创建Matplotlib图像
fig, axes = plt.subplots(3, 8, figsize=(16, 6))

# 使用列表推导式加载和显示图像
for ax, img_file in zip(axes.flat, image_files):
    img_path = os.path.join(image_folder, img_file)
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')

# 显示图像
plt.tight_layout()
plt.show()

# step4. 数据预处理
total_datadir = './weather/'

# 关于transforms.Compose的更多介绍可以参考：https://blog.csdn.net/qq_38251616/article/details/124878863
train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    transforms.ToTensor(),          # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    transforms.Normalize(           # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
])

total_data = datasets.ImageFolder(total_datadir, transform=train_transforms)
print(f"{total_data=}")

imgs, labels = next(iter(total_data))
print(f"after wrap: imgs.shape={imgs.shape}, labels={labels}")

train_size = int(0.8 * len(total_data))
test_size = len(total_data) - train_size
train_data, test_data = torch.utils.data.random_split(total_data, [train_size, test_size])
print(f"{train_data=}, {test_data=}")

train_dl = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

for X, y in test_dl:
    print(f"X.shape [N, C, W, H]={X.shape}")
    print(f"y.shape={y.shape}")
    break


# step5. 模型定义
import torch.nn.functional as F

class Network_bn(nn.Module):
    def __init__(self):
        super(Network_bn, self).__init__()
        """
        nn.Conv2d()函数：
        第一个参数（in_channels）是输入的channel数量
        第二个参数（out_channels）是输出的channel数量
        第三个参数（kernel_size）是卷积核大小
        第四个参数（stride）是步长，默认为1
        第五个参数（padding）是填充大小，默认为0
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(12)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*50*50, len(classeNames))

    def forward(self, x):
        x = self.dropout(self.bn1(self.conv1(x)))
        x = F.relu(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = x.view(-1, 24*50*50)
        x = self.fc1(x)

        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = Network_bn().to(device)

from torchsummary import summary
summary(model, (3, 224, 224), 32)

# step6. 损失函数和优化器 学习率 等 超参数配置
loss_fn = nn.CrossEntropyLoss() # 创建损失函数
learn_rate = 1e-4 # 学习率
opt = torch.optim.SGD(model.parameters(),lr=learn_rate, momentum=0.5)

scheduler = ExponentialLR(opt, gamma=0.9)

# step7. 训练模型

'''
**1. optimizer.zero_grad()**

函数会遍历模型的所有参数，通过内置方法截断反向传播的梯度流，再将每个参数的梯度值设为0，即上一次的梯度记录被清空。

**2. loss.backward()**

PyTorch的反向传播(即`tensor.backward()`)是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。

具体来说，torch.tensor是autograd包的基础类，如果你设置tensor的requires_grads为True，就会开始跟踪这个tensor上面的所有运算，如果你做完运算后使用`tensor.backward()`，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。

更具体地说，损失函数loss是由模型的所有权重w经过一系列运算得到的，若某个w的requires_grads为True，则w的所有上层参数（后面层的权重w）的.grad_fn属性中就保存了对应的运算，然后在使用`loss.backward()`后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。

如果没有进行`tensor.backward()`的话，梯度值将会是None，因此`loss.backward()`要写在`optimizer.step()`之前。

**3. optimizer.step()**

step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以在执行`optimizer.step()`函数前应先执行`loss.backward()`函数来计算梯度。

注意：optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是`tensor.backward()`方法产生的。
'''
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

# step8. 测试模型
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

# step9. 训练模型
'''
1. model.train()

model.train()的作用是启用 Batch Normalization 和 Dropout。

如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。

2. model.eval()

model.eval()的作用是不启用 Batch Normalization 和 Dropout。

如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有BN层和Dropout所带来的的性质。
'''

epochs = 5
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    model.train()
    epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_fn, opt)

    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_fn)

    scheduler.step()

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    lr = opt.state_dict()['param_groups'][0]['lr']

    template = (
        'Epoch:{:2d}, Learn_Rate: {:e}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%，Test_loss:{:.3f}')
    print(
        template.format(epoch + 1, lr, epoch_train_acc * 100, epoch_train_loss, epoch_test_acc * 100, epoch_test_loss))
print('Done')

# step10. 可视化训练过程
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