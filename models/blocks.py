import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, n_in, n_middle, n_out):
        super(ConvBlock, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.conv1 = nn.Conv2d(n_in, n_middle, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(n_middle)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(n_middle, n_out, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(n_out)

        if n_in != n_out:
            self.identity = nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0)
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        return x + identity


class DownSampleLayer(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=4, stride=2, padding=1):
        super(DownSampleLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(n_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class ResBlockUp(nn.Module):
    def __init__(self, n_in, n_middle, n_out):
        super(ResBlockUp, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_middle, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_middle),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n_middle, n_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_out),
        )

        self.identity = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv(x)

        return x + identity


class ResBlockDown(nn.Module):
    def __init__(self, n_in, n_middle, n_out):
        super(ResBlockDown, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_middle, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_middle),
            nn.SiLU(inplace=True),
            nn.Conv2d(n_middle, n_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_out),
            nn.AvgPool2d(2),
        )

        self.identity = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv(x)

        return x + identity
