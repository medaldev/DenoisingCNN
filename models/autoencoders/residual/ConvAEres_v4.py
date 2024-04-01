from torch import nn
from models.autoencoders.residual import AEv2_0


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)

class AEv4_0(nn.Module):
    def __init__(self, in_channels, out_channels, fm_path):
        super(AEv4_0, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.first = torch.load(fm_path)

        for param in self.first.parameters():
            param.requires_grad = False

        self.main = AEv2_0(2, out_channels)
        self.main.apply(weights_init)


    def forward(self, x):
        # ========== encoder ==========
        inp = x
        x = self.first(x)
        x = self.main(torch.cat([inp, x], dim=1))

        return x


import torch

if __name__ == '__main__':
    device = torch.device("cpu:0")
    model = AEv4_0(1, 1, "/home/amedvedev/projects/python/DenoisingCNN/assets/pt/uvych_matrix_denoiser_10 (4th copy).pt").to(device).eval()
    x = torch.randn(4, 1, 64, 64, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
