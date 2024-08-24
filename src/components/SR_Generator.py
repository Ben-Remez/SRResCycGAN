import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.prelu(self.conv1(x)))


class GSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residuals=5):
        super(GSR, self).__init__()
        self.encoder = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.resnet = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residuals)])
        self.decoder = nn.Conv2d(64, out_channels, kernel_size=5, padding=2)
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

    def forward(self, x):
        x = self.upsample(x)
        x = self.encoder(x)
        x = self.resnet(x)
        x = self.decoder(x)
        return torch.clamp(x, 0, 1)