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


    def forward(self, x):
        # x = torch.fft.fftn(x, dim=(-3, -2, -1))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = complex_relu(x)
        # x = torch.fft.ifftn(x, dim=(-3, -2, -1))
        return x


if __name__ == '__main__':
    k_uvych = 16
    device = torch.device("cpu")
    dtype = torch.complex64
    model = Encoder(channels=[3, 32, 64, 128, 64, 32, 3],
                    kernel_size=2, stride=1, padding=1).to(device, dtype=dtype)
    model.eval()

    # x = torch.randn(5, 3, 10, 10, 10, device=device, dtype=dtype)
    x = torch.from_numpy(load("D:/projects/gcggenE/InverseProblemE/DATA/tasks/5/Evych").reshape(1, 3, 10, 10, 10)).to(
        device, dtype)

    with torch.no_grad():
        r = model(x)
        print(r)
        print(r.size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
