from torch import nn


class ConvAutoencoderLumaRelu10(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvAutoencoderLumaRelu10, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels


        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=in_channels, out_channels=76, kernel_size=10, stride=2, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=76, out_channels=38, kernel_size=7, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=38, out_channels=38, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=38, out_channels=38, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=38, out_channels=38, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=38, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=38, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=38, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.latent = nn.Sequential(
            nn.Linear(20 * 20, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 20 * 20),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            # Decode

            nn.ConvTranspose2d(in_channels=1, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=38, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=38, out_channels=76, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=76, out_channels=76, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=76, out_channels=out_channels, kernel_size=3, stride=1, padding=0),


        )

    def forward(self, x):
        bs = x.size()[0]
        x = self.encoder(x)
        x = self.latent(x.view(bs, 20 * 20)).view(bs, 1, 20, 20)
        x = self.decoder(x)

        return x


import torch
if __name__ == '__main__':
    device = torch.device("cpu:0")
    model = ConvAutoencoderLumaRelu10(64,1).to(device).eval()
    x = torch.randn(4, 64, 80, 80, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))

