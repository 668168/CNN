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
