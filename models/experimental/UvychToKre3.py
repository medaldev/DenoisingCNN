import torch
from torch import nn
import numpy as np
from .UvychToKre2 import UvychToKre2

class UvychToKre3(nn.Module):
    def __init__(self):
        super(UvychToKre3, self).__init__()

        self.block1 = UvychToKre2()
        self.block2 = UvychToKre2()

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm2d(num_features=900)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


if __name__ == '__main__':
    import torch
    m = UvychToKre3().eval()
    x = torch.randn(1, 30, 30)
    print(m(x).size())
    print(sum(p.numel() for p in m.parameters() if p.requires_grad))