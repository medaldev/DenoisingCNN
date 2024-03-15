from torch import nn
import torch

class ConvTwoToOne(nn.Module):
    def __init__(self):
        super(ConvTwoToOne, self).__init__()

        self.all_layers = nn.Sequential(

            # Encode

            nn.Conv2d(in_channels=1, out_channels=76, kernel_size=3, stride=1, padding=1),
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
            nn.ConvTranspose2d(in_channels=76, out_channels=1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid()

        )

    def forward(self, x1, x2):
        x = self.all_layers(x1) + self.all_layers(x2)

        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    model = ConvTwoToOne().to(device)
    inp1 = torch.randn((1, 1, 100, 100), device=device)
    inp2 = torch.randn((1, 1, 100, 100), device=device)

    with torch.no_grad():
      print(model(inp1, inp2).size())
