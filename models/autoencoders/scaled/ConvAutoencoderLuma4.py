from torch import nn
import torch


class ConvAutoencoderLuma4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvAutoencoderLuma4, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.Sequential(

            # Encode

            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
        )

        self.decoder = nn.Sequential(

            # Decode

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=self.out_channels, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = ConvAutoencoderLuma4(in_channels=1, out_channels=1).to(device).eval()
    x = torch.randn(1, 1, 30, 30, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
