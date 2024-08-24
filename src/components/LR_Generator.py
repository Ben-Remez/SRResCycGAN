import torch.nn as nn


class GLR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GLR, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
             self._block(64, 128, 3, 2),
             self._block(128, 256, 3, 2),
            nn.Conv2d(256, out_channels, kernel_size=3, padding=1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.net(x)