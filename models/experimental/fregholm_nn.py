import torch
from torch import nn
import numpy as np

class Phi(nn.Module):
    def __init__(self):
        super(Phi, self).__init__()

        self.ln1 = nn.Linear(1, 10)
        self.ln2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.ln1(x)
        x = nn.ReLU(x)
        x = self.ln2(x)
        x = nn.ReLU(x)
        return x


class FredholmIntegral(nn.Module):
    def __init__(self, K):
        super(FredholmIntegral, self).__init__()
        self.phi = Phi()
        self.K = K

    def forward(self, x, a, b, n, lam):
        int_sum = torch.zeros(1, requires_grad=True)
        h = (b - a) / n
        for i in range(n):
            s = a + h * (i + 0.5)
            int_sum += self.phi(s) * self.K(x, s)

        return lam * int_sum * h

if __name__ == '__main__':

    def Tn(n, x):
        return np.cos(n * np.arccos(x))

    def K(x, y):
        return (1. / abs(x - y) + x * y) / (1. / (1 - x ** 2) ** 0.5)

    def f(y):
        return 1. + y

    def phi_orig(x):
        1 / np.pi / np.log(2) * Tn(0, x) + 2 / 3 / np.pi * Tn(1, x)

    int_op = FredholmIntegral(K)

    print(int_op(0.3, -1.0, 1.0, 10, 0.5))


