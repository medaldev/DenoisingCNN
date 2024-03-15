from torch import nn
import torch


class PredictRealMaxMin(nn.Module):
    def __init__(self, in_channels):
        super(PredictRealMaxMin, self).__init__()

        self.in_channels = in_channels

        self.encoder = nn.Sequential(

            # Encode

            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
        )

        self.predictor = nn.Sequential(

            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=2)

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = PredictRealMaxMin(in_channels=1).to(device).eval()
    x = torch.randn(1, 1, 80, 80, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
