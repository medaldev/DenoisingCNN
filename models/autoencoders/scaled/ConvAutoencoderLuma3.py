from torch import nn


class ConvAutoencoderLuma3(nn.Module):
    def __init__(self):
        super(ConvAutoencoderLuma3, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=76, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=76, out_channels=50, kernel_size=2, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(in_channels=50, out_channels=75, kernel_size=2, stride=1, padding=1,
                                          output_padding=0)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=75, out_channels=1, kernel_size=2, stride=1, padding=1,
                                          output_padding=0)

        self.all_layers = nn.Sequential(

            # Encode

            nn.Conv2d(in_channels=1, out_channels=76, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1),

            nn.Conv2d(in_channels=76, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1),

            nn.Conv2d(in_channels=38, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1),

            # Decode

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=38, out_channels=76, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=76, out_channels=1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.all_layers(x)

        return x


if __name__ == '__main__':
    import torch
    m = ConvAutoencoderLuma3().eval()
    x = torch.randn(1, 1, 100, 100)
    print(m(x).size())