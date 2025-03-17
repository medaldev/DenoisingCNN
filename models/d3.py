import numpy as np
import torch
from torch import nn


def complex_relu(z):
    return torch.relu(z.real) + 1j * torch.relu(z.imag)


class Encoder(nn.Module):
    def __init__(self, channels: list[int], kernel_size, stride, padding):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(in_channels=channels[i - 1], out_channels=channels[i], kernel_size=kernel_size,
                      stride=stride, padding=padding)

            for i in range(1, len(channels))
        ])
        self.flatten = nn.Flatten()
        self.linears = nn.ModuleList([
            nn.Linear(4096, 512),
            nn.Linear(512, 5),
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = complex_relu(x)
        print("lc", x.size())
        x = self.flatten(x)
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < len(self.linears) - 1:
                x = complex_relu(x)
        x = torch.abs(x)
        return x


def load(path):
    re_im = np.fromfile(path,  dtype=np.double)
    x = np.vectorize(complex)(re_im[::2], re_im[1::2])
    return x


if __name__ == '__main__':
    k_uvych = 16
    device = torch.device("cpu")
    dtype = torch.cdouble
    model = Encoder(channels=[1, 32, 64, 128, 256, 512, 512, 256, 128, 64, 32, 1],
                    kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)).to(device, dtype=dtype)
    model.eval()

    #x = torch.randn(5, 1, 16, 16, 16, device=device, dtype=dtype)
    x = torch.from_numpy(load("D:\\projects\\cpp\\gcggen\\data\\examples\\0\\K").reshape(1, 1, 16, 16, 16)).to(device, dtype)

    with torch.no_grad():
        r = model(x)
        print(r)
        print(r.size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
