import torch
from torch import nn
import numpy as np

class UvychToKre(nn.Module):
    def __init__(self):
        super(UvychToKre, self).__init__()

        self.ln1 = nn.Linear(900, 2048)
        self.ln2 = nn.Linear(2048, 2048)
        self.ln3 = nn.Linear(2048, 2048)
        self.ln4 = nn.Linear(2048, 900)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size()[0], 900)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.ln3(x)
        x = self.relu(x)
        x = self.ln4(x)
        x = self.relu(x)
        x = x.view(x.size()[0], 30, 30)
        return x


if __name__ == '__main__':
    import torch
    m = UvychToKre().eval()
    x = torch.randn(1, 1, 30, 30)
    print(m(x).size())