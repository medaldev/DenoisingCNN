from torch import nn
import torch

class ConvAutoencoderLumaRelu5(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=10, n_linears=1):
        super(ConvAutoencoderLumaRelu5, self).__init__()

        self.convs = []
        self.tconvs = []
        self.convs.append(
            nn.Conv2d(in_channels=in_channels, out_channels=2 ** (0 + 3), kernel_size=2, stride=2, padding=0))

        self.n = n_convs
        for i in range(n_convs):
            self.convs.append(nn.Conv2d(in_channels=min(2**(i + 3), 2 ** 11), out_channels=min(2**(i + 4), 2 ** 11),
                                        kernel_size=4, stride=1, padding=0))

            j = n_convs - i - 1
            self.tconvs.append(nn.ConvTranspose2d(in_channels=min(2 ** (j + 4), 2 ** 11), out_channels=min(2 ** (j + 3), 2 ** 11),
                                        kernel_size=4, stride=1, padding=0))
        self.tconvs.append(
            nn.ConvTranspose2d(in_channels=min(2 ** (j + 3), 2 ** 11), out_channels=out_channels, kernel_size=2, stride=2, padding=0))


        self.linears = []
        for i in range(n_linears):
            self.linears.append(nn.Linear(8192, 8192))

        self.convs = nn.ModuleList(self.convs)
        self.tconvs = nn.ModuleList(self.tconvs)
        self.linears = nn.ModuleList(self.linears)

        self.flatten1 = nn.Flatten()


        self.lrelu = nn.LeakyReLU()
        self.dp = nn.Dropout()



    def forward(self, x):
        bs = x.size()[0]

        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.lrelu(x)

        x = self.flatten1(x)

        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.lrelu(x)

        x = x.view(bs, 2048, 2, 2)

        for i in range(len(self.tconvs)):
            x = self.tconvs[i](x)
            if i < len(self.tconvs) - 1:
                x = self.lrelu(x)

        return x


if __name__ == '__main__':
    device = torch.device("cpu:0")
    model = ConvAutoencoderLumaRelu5(2, 1, 10, 4).to(device).eval()
    x = torch.randn(1, 2, 64, 64, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
