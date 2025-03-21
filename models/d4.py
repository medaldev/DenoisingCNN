import numpy as np
import torch
from torch import nn


import numpy as np
import torch
from torch import nn


def complex_relu(z):
    return torch.relu(z.real) + 1j * torch.relu(z.imag)


class FilterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_fft: bool = False):
        super(FilterBlock, self).__init__()
        self.use_fft = use_fft
        self.layers = nn.ModuleList([
            nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),


        ])

    def forward(self, x):
        if self.use_fft:
            x = torch.fft.fftn(x, dim=(-3, -2, -1))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = complex_relu(x)
        if self.use_fft:
            x = torch.fft.ifftn(x, dim=(-3, -2, -1))
        return x


class DenoiserBlock(nn.Module):
    def __init__(self, in_channels, num_par_filters, out_channels):
        super(DenoiserBlock, self).__init__()
        self.num_par_filters = num_par_filters
        self.par_fil = nn.ModuleList([
            FilterBlock(in_channels=in_channels, out_channels=3, use_fft=(i % 2) == 0)
            for i in range(num_par_filters)
        ])
        self.reducer = FilterBlock(in_channels=3 * num_par_filters, out_channels=out_channels)

    def forward(self, x):
        par_results = [self.par_fil[i](x) for i in range(self.num_par_filters)]
        stacked = torch.stack(par_results)
        regrouped_tensor = stacked.transpose(0, 1)
        regr_size = list(regrouped_tensor.size())
        regrouped_tensor = regrouped_tensor.resize(regr_size[0], regr_size[1] * regr_size[2], *regr_size[3:])
        reduced_tensor = self.reducer(regrouped_tensor)
        return reduced_tensor


class DenoiserModel(nn.Module):
    def __init__(self, in_channels, num_par_filters, num_denoiser_blocks):
        super(DenoiserModel, self).__init__()
        self.num_denoiser_blocks = num_denoiser_blocks
        self.blocks = []
        for i in range(num_denoiser_blocks):
            self.blocks.append(
                DenoiserBlock(in_channels, num_par_filters, 3 if i + 1 == num_denoiser_blocks else in_channels)
            )
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        for i in range(self.num_denoiser_blocks):
            x = self.blocks[i](x)
        return x



def load(path):
    re_im = np.fromfile(path, dtype=np.double)
    x = np.vectorize(complex)(re_im[::2], re_im[1::2])
    return x


if __name__ == '__main__':
    k_uvych = 16
    device = torch.device("cpu")
    dtype = torch.complex64
    model = FilterBlock(3, 3).to(device, dtype=dtype)
    model.eval()

    x = torch.randn(5, 3, 10, 10, 10, device=device, dtype=dtype)

    with torch.no_grad():
        r = model(x)
        print(r)
        print(r.size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
