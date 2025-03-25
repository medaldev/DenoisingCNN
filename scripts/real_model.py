import numpy as np
import torch
from torch import nn
from torch.nn.functional import batch_norm

import torch
import torch.nn as nn


class FilterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_fft: bool = False):
        super(FilterBlock, self).__init__()
        self.use_fft = use_fft

        # Define convolutional layers with batch normalization
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(64)
        self.bn5 = nn.BatchNorm3d(64)
        self.bn6 = nn.BatchNorm3d(32)
        self.bn7 = nn.BatchNorm3d(16)
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv3d(16, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if self.use_fft:
            x = torch.fft.fftn(torch.view_as_complex(x), dim=(-3, -2, -1))

        # Apply layers with residual connections and batch normalization
        x1_res = self.conv1(x)
        x1 = torch.relu(self.bn1(x1_res))

        x2_res = self.conv2(x1)
        x2 = torch.relu(self.bn2(x2_res))

        x3_res = self.conv3(x2)
        x3 = torch.relu(self.bn3(x3_res))

        x4_res = self.conv4(x3)
        x4 = torch.relu(self.bn4(x4_res))

        x5_res = self.conv5(x4)
        x5 = torch.relu(self.bn5(x5_res)) + x3_res

        x6_res = self.conv6(x5)
        x6 = torch.relu(self.bn6(x6_res)) + x2_res

        x7_res = self.conv7(x6)
        x7 = torch.relu(self.bn7(x7_res)) + x1_res

        x8 = self.conv8(x7)

        if self.use_fft:
            x8 = torch.view_as_real(torch.fft.ifftn(x8, dim=(-3, -2, -1)))

        return x8


class DenoiserBlock(nn.Module):
    def __init__(self, in_channels, num_par_filters, out_channels):
        super(DenoiserBlock, self).__init__()
        self.num_par_filters = num_par_filters
        self.par_fil = nn.ModuleList([
            FilterBlock(in_channels=in_channels, out_channels=3)
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


if __name__ == '__main__':
    k_uvych = 16
    device = torch.device("cpu")
    dtype = torch.double
    # model = DenoiserModel(in_channels=90, num_par_filters=5, num_denoiser_blocks=2).to(device, dtype=dtype)
    model = FilterBlock(in_channels=90, out_channels=3).to(device, dtype=dtype)
    model.eval()

    x = torch.randn(6, 90, 10, 10, 10, device=device, dtype=dtype)
    # x = torch.from_numpy(load("D:/projects/gcggenE/InverseProblemE/DATA/tasks/5/Evych").reshape(1, 3, 10, 10, 10)).to(
    #     device, dtype)

    with torch.no_grad():
        r = model(x)
        print(r.size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
