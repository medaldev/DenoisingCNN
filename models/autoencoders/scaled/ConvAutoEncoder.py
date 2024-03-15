from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.all_layers = nn.Sequential(

            # Encode

            nn.Conv2d(in_channels=3, out_channels=76, kernel_size=3, stride=1, padding=1),
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
            nn.ConvTranspose2d(in_channels=76, out_channels=3, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.all_layers(x)

        return x



