from torch import nn


class AEv2_0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AEv2_0, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_transpose0 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_transpose4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_transpose5 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_transpose6 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_transpose7 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # ========== encoder ==========
        x = self.conv0(x)
        x = self.relu(x)
        res0 = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        res1 = x
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        res2 = x
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        res3 = x
        x = self.conv7(x)
        x = self.relu(x)
        # ========== decoder ==========
        x = self.conv_transpose0(x)
        x = self.relu(x)
        x += res3
        x = self.conv_transpose1(x)
        x = self.relu(x)
        x = self.conv_transpose2(x)
        x = self.relu(x)
        x += res2
        x = self.conv_transpose3(x)
        x = self.relu(x)
        x = self.conv_transpose4(x)
        x = self.relu(x)
        x += res1
        x = self.conv_transpose5(x)
        x = self.relu(x)
        x = self.conv_transpose6(x)
        x = self.relu(x)
        x += res0
        x = self.conv_transpose7(x)
        return x


import torch

if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = AEv2_0(1, 1).to(device).eval()
    x = torch.randn(4, 1, 64, 64, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
