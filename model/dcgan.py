import os
import argparse
import numpy as np
import torch
from torch import nn, optim

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image
from torchvision import datasets, transforms


def get_hidden_list(n, h, n_layers, alpha):
    return [int(np.round(n - (n-h)*(x/n_layers)**alpha)) for x in np.arange(n_layers)]


class DCDiscriminator(nn.Module):
    def __init__(self, inp_size: int, image_bottleneck: int,
                    n_img_layers: int, labels_dim: int,
                    alpha: float = 1.):
        assert isinstance(inp_size, int) and int > 0

        super(ModelD, self).__init__()
        x_layers_list = []
        for insize_, outsize_ in zip(self.encoder_sizes, self.encoder_sizes[1:]):
            x_layers_list.extend([
                Linear(insize_, outsize_),
                MishLayer(),
                BatchNorm1d(outsize_),
            ])

        self.x_pipeline = nn.Sequential(*x_layers_list)
        self.conditional_pipeline = nn.Sequential(
            nn.Linear(image_bottleneck + labels_dim, (image_bottleneck + labels_dim) // 2),
            MishLayer(),
            BatchNorm1d((image_bottleneck + labels_dim) // 2),
        )

        self.y_linear = nn.Sequential(
            nn.Linear(labels_dim, max(inp_size // 10, labels_dim + 1)),
            MishLayer(),
            BatchNorm1d((image_bottleneck + labels_dim) // 2),
        )
        self.fc = nn.Linear((image_bottleneck + labels_dim) // 2, 1)

    def forward(self, x, labels):
        x = self.x_pipeline(x)
        y_ = self.y_linear(labels)
        x = torch.cat([x, y_], 1)
        x = self.conditional_pipeline(x)
        x = self.fc(x)
        return F.sigmoid(x)

class ModelG(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelG, self).__init__()
        self.fc2 = nn.Linear(10, 1000)
        self.fc = nn.Linear(self.z_dim+1000, 64*28*28)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc2(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn1(x) 
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.sigmoid(x)
        return x