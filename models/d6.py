import numpy as np
import torch
from torch import nn
from complextorch.nn.modules.conv import CVConv3d
from complextorch.nn.modules.batchnorm import CVBatchNorm3d
from complextorch.nn.modules.pooling import CVAdaptiveAvgPool3d


def complex_relu(z):
    return torch.relu(z.real) + 1j * torch.relu(z.imag)


class FilterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_fft: bool = False, device=None, dtype=None):
        super(FilterBlock, self).__init__()
        self.use_fft = use_fft

        # Complex Convolutional Layer
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding)

        # Complex Batch Normalization
        self.bn = CVBatchNorm3d(out_channels)

        # Complex Adaptive Average Pooling
        self.pool = CVAdaptiveAvgPool3d(output_size=(1, 1, 1))

    def forward(self, x):
        if self.use_fft:
            x = torch.fft.fftn(x, dim=(-3, -2, -1))

        # Apply Complex Convolution
        x = self.conv(x)

        # Apply Complex Batch Normalization
        x = self.bn(x)

        # Apply Complex ReLU Activation
        x = complex_relu(x)

        # Apply Complex Adaptive Average Pooling
        x = self.pool(x)

        if self.use_fft:
            x = torch.fft.ifftn(x, dim=(-3, -2, -1))

        return x

class DenoiserBlock(nn.Module):
    def __init__(self, in_channels, num_par_filters):
        super(DenoiserBlock, self).__init__()
        self.num_par_filters = num_par_filters
        self.par_fil = nn.ModuleList([
            FilterBlock(in_channels=in_channels, out_channels = 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                        use_fft=(i % 2) == 0)
            for i in range(num_par_filters)
        ])
        self.reducer = FilterBlock(in_channels=in_channels * num_par_filters,out_channels=in_channels,
                                   kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

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
                DenoiserBlock(in_channels, num_par_filters)
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

    #model = FilterBlock(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device).to(device, dtype=dtype)
    model = FilterBlock(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device).to(device, dtype=dtype)
    model.eval()

    x = torch.randn(6, 3, 10, 10, 10, device=device, dtype=dtype)

    with torch.no_grad():
        r = model(x)
        print(r.size())
    #
    # model = DenoiserModel(in_channels=3, num_par_filters=5, num_denoiser_blocks=2).to(device, dtype=dtype)
    # model.eval()
    #
    # x = torch.randn(6, 3, 10, 10, 10, device=device, dtype=dtype)
    # # x = torch.from_numpy(load("D:/projects/gcggenE/InverseProblemE/DATA/tasks/5/Evych").reshape(1, 3, 10, 10, 10)).to(
    # #     device, dtype)
    #
    # with torch.no_grad():
    #     r = model(x)
    #     print(r.size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
