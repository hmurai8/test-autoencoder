import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch import nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# for reproducibility
torch.manual_seed(0)

class AutoEncoder:
    def __init__(self):
        super(AutoEncoder, self).__init__()


        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) 


# dataset pre-processing
train_dataset = datasets.CIFAR10(root='./data',\
              train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, \
                                batch_size=32, shuffle=True)