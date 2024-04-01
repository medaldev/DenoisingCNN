from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class DeConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2):
        super(DeConvBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class AEv6_0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AEv6_0, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 80, 2),
            ConvBlock(80, 80, 3),
            ConvBlock(80, 80, 2),
            ConvBlock(80, 80, 3),
        )

        self.decoder = nn.Sequential(
            DeConvBlock(80, 80, 3),
            DeConvBlock(80, 80, 2),
            DeConvBlock(80, 80, 3),
            DeConvBlock(80, out_channels, 2),
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # ========== encoder ==========
        x = self.encoder(x)
        x = self.relu(x)
        res0 = x
        x = self.conv1(x)
        x = self.relu(x)
        res1 = x
        x = self.conv2(x)
        x = self.relu(x)
        res2 = x
        x = self.conv3(x)
        x = self.relu(x)
        res3 = x
        x = self.conv4(x)
        x = self.relu(x)
        res4 = x
        x = self.conv5(x)
        x = self.relu(x)
        res5 = x
        x = self.conv6(x)
        x = self.relu(x)
        res6 = x
        x = self.conv7(x)
        x = self.relu(x)
        res7 = x
        x = self.conv8(x)
        x = self.relu(x)
        res8 = x
        x = self.conv9(x)
        x = self.relu(x)
        res9 = x
        x = self.conv10(x)
        x = self.relu(x)

        # ========== decoder ==========
        x = self.conv_transpose0(x)
        x = self.relu(x)
        x += res9
        x = self.conv_transpose1(x)
        x = self.relu(x)
        x += res8
        x = self.conv_transpose2(x)
        x = self.relu(x)
        x += res7
        x = self.conv_transpose3(x)
        x = self.relu(x)
        x += res6
        x = self.conv_transpose4(x)
        x = self.relu(x)
        x += res5
        x = self.conv_transpose5(x)
        x = self.relu(x)
        x += res4
        x = self.conv_transpose6(x)
        x = self.relu(x)
        x += res3
        x = self.conv_transpose7(x)
        x = self.relu(x)
        x += res2
        x = self.conv_transpose8(x)
        x = self.relu(x)
        x += res1
        x = self.conv_transpose9(x)
        x = self.relu(x)
        x += res0
        x = self.conv_transpose10(x)
        return x


import torch

if __name__ == '__main__':
    device = torch.device("cpu:0")
    model = AEv6_0(1, 1).to(device).eval()
    x = torch.randn(4, 1, 80, 80, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
