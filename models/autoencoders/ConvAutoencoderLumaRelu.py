from torch import nn


class ConvAutoencoderLumaRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvAutoencoderLumaRelu, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Encoder

        self.all_layers = nn.Sequential(

            # Encode

            nn.Conv2d(in_channels=in_channels, out_channels=76, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=76, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=38, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),

            # Decode

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=38, out_channels=76, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=76, out_channels=out_channels, kernel_size=2, stride=1, padding=0),
            nn.ReLU()

        )

    def forward(self, x):
        x = self.all_layers(x)

        return x


import torch

if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = ConvAutoencoderLumaRelu(64, 1).to(device).eval()
    x = torch.randn(4, 64, 30, 30, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
