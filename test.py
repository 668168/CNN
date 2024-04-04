import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from torchvision import datasets
from torchvision.transforms import ToTensor

import math
import numpy as np


lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for
batch_size = 64  # batch size

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# train_ds = TensorDataset(x_train, y_train)
train_dl = test_dataloader
