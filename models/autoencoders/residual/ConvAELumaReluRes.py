from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class ConvAELumaReluRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvAELumaReluRes, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Encoder

        self.c1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=0)
        self.c4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=0)

        self.uc1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=1, padding=0)
        self.uc2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=1, padding=0)
        self.uc3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.uc4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=1, padding=0)

        self.uc_res = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = c1_res = F.relu(self.c1(x))
        x = c2_res = F.relu(self.c2(x))

        x = c3_res = F.relu(self.c3(x))
        x = F.relu(self.c4(x))


        x = F.relu(self.uc1(x)) + c3_res
        x = F.relu(self.uc2(x)) + c2_res
        x = F.relu(self.uc3(x)) + c1_res
        x = F.relu(self.uc4(x))
        x = self.uc_res(x)

        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


import torch

if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = ConvAELumaReluRes(1, 1).to(device).eval()
    model.apply(init_weights)
    x = torch.randn(4, 1, 64, 64, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
