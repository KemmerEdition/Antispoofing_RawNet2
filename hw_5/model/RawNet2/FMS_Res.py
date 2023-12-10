import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FMS(nn.Module):
    def __init__(self, dimens):
        super().__init__()

        self.linear = nn.Linear(in_features=dimens, out_features=dimens)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.pooling(x).reshape(x.shape[0], -1)
        res = self.linear(res)
        res = self.sigmoid(res).reshape(x.shape[0], x.shape[1], -1)
        x = x * res + res
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        list_layers = []
        if not first:
            list_layers += [nn.BatchNorm1d(in_channels), nn.LeakyReLU(negative_slope=0.3)]
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)
        list_layers += [nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm1d(out_channels),
                        nn.LeakyReLU(negative_slope=0.3),
                        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)]
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.list_layers = nn.Sequential(*list_layers)
        if in_channels != out_channels:
            self.flag = True
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        else:
            self.flag = False
        self.polling = nn.MaxPool1d(3)
        self.fms_blok = FMS(out_channels)

    def forward(self, x):
        # if not self.first:
        #     res = self.leaky_relu(self.bn1(x))
        res = self.list_layers(x)
        # res = self.conv(res)
        # res = self.conv2(self.leaky_relu(self.bn2(res)))
        if self.flag:
            x = self.proj(x)
        res = res + x
        res = self.fms_blok(self.polling(res))
        return res
