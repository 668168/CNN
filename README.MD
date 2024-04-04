

## 维度缩减
`npimg = np.squeeze(imgs.numpy())` # 从(1, 28, 28) -> (28, 28)

`squeeze()`函数的功能是从矩阵shape中，去掉维度为1的。例如一个矩阵是的shape是（5, 1），
使用过这个函数后，结果为（5, ）。

## CHW -> HWC transpose
NumPy表示的RGB图像，形状为(高度, 宽度, 通道数)，即HWC格式。
PyTorch的图像数据张量形状为(批量大小, 通道数, 高度, 宽度)，即NCHW格式。
numpy 和 PyTorch 的通道顺序都是 R-G-B
transpose 把PyTorch 的CHW格式(1, 28, 28) 转为 numpy 的HWC格式 (28, 28, 1)
也就是说把第一个维度放到最后一个维度
numpy 的transpose函数可以实现这个功能
虽然在NumPy中处理图像时通常使用(高度, 宽度, 通道数)的排列顺序，但在某些深度学习框架中（如PyTorch），
图像数据通常以(通道数, 高度, 宽度)的形式表示。这种差异主要是由于不同库和框架的约定不同，因此在使用时需要注意转换。
`npimg = imgs.numpy().transpose((1, 2, 0))`


## argmax(1) 表示取第二个维度的最大值的下标. 比如A11, A21, A31,..., A64-1 组合为一个向量，取最大值的下标
1 表示 1 的下标，即第二个维度，即为预测值.
item 把一个标量Tensor转换为一个Python number
`train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()`


## model.eval()

model.eval()的作用是不启用 Batch Normalization 和 Dropout。

如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。
model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，
否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有BN层和Dropout所带来的的性质。


## torch.flatten(x, start_dim=1)

`torch.flatten()`函数的作用是将输入张量x按照start_dim维度展平。


## optimistic Adam, SGD, momentum
神奇, SGD的准确率到94%, 但Adam 的准确率只有11%.
SGD 参数加了momentum=0.9, 效果很随机. 移除后正常.可以作为案例面试时候分享.
动量（Momentum）的工作原理：
动量方法借鉴了物理中的“动量”概念，模拟了一个球在有摩擦的表面上滚动的过程。
在这个过程中，球不仅受到当前梯度的影响，也会保留之前速度的一部分，
这样可以在某个方向上积累速度，从而加快收敛速度并帮助跳出局部最小值。
```python
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
opt = optim.SGD(model.parameters(), lr=lr)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
```

## H_out = (H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1

dilation：控制窗口中元素步幅的参数

