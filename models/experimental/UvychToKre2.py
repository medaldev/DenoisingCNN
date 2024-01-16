import torch
from torch import nn
import numpy as np

class UvychToKre2(nn.Module):
    def __init__(self):
        super(UvychToKre2, self).__init__()

        self.ln1 = nn.Linear(900, 1024)
        self.ln2 = nn.Linear(1024, 2048)
        self.ln3 = nn.Linear(2048, 4096)
        self.ln4 = nn.Linear(4096, 8192)
        self.ln5 = nn.Linear(8192, 4096)
        self.ln6 = nn.Linear(4096, 2048)
        self.ln7 = nn.Linear(2048, 1024)
        self.ln8 = nn.Linear(1024, 900)
        self.ln9 = nn.Linear(900, 900)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.bn2 = nn.BatchNorm1d(num_features=2048)
        self.bn3 = nn.BatchNorm1d(num_features=4096)
        self.bn4 = nn.BatchNorm1d(num_features=8192)

    def forward(self, x):
        x = x.view(x.size()[0], 900)

        x = self.ln1(x)
        x = self.bn1(x)
        x1 = self.selu(x)

        x = self.ln2(x1)
        x = self.bn2(x)
        x2 = self.selu(x)

        x = self.ln3(x2)
        x3 = self.selu(x)

        x = self.ln4(x3)
        x = self.bn4(x)
        x4 = self.selu(x)

        x = self.ln5(x4)
        x5 = self.selu(x) + x3

        x = self.ln6(x5)
        x = self.bn2(x)
        x6 = self.selu(x) + x2

        x = self.ln7(x6)
        x7 = self.selu(x) + x1

        x = self.ln8(x7)
        x8 = self.selu(x) + x

        x = self.ln9(x8)
        x = x.view(x.size()[0], 1, 30, 30)
        return x


if __name__ == '__main__':
    import torch
    m = UvychToKre2().eval()
    x = torch.randn(1, 30, 30)
    print(m(x).size())