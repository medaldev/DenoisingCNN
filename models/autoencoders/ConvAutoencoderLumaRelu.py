from torch import nn


class ConvAutoencoderLumaRelu(nn.Module):
    def __init__(self):
        super(ConvAutoencoderLumaRelu, self).__init__()

        # Encoder


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
            nn.ReLU()

        )

    def forward(self, x):
        x = self.all_layers(x)

        return x
