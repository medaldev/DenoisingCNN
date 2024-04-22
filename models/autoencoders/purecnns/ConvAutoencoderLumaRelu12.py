from torch import nn
import torch

class ConvAutoencoderLumaRelu12(nn.Module):
    def __init__(self, convs_ch=None, linears_ch=None):
        super(ConvAutoencoderLumaRelu12, self).__init__()

        if linears_ch is None:
            linears_ch = []
        if convs_ch is None:
            convs_ch = [1, 1]

        self.convs = []
        self.tconvs = []


        n_convs = len(convs_ch)

        for i in range(len(convs_ch) - 1):
            self.convs.append(nn.Conv2d(in_channels=convs_ch[i], out_channels=convs_ch[i + 1],
                                        kernel_size=4, stride=1, padding=0))

            self.tconvs.append(nn.ConvTranspose2d(in_channels=convs_ch[n_convs - i - 1], out_channels=convs_ch[n_convs - i - 2],
                                        kernel_size=4, stride=1, padding=0))

        self.linears = []
        for i in range(len(linears_ch) - 1):
            self.linears.append(nn.Linear(linears_ch[i], linears_ch[i + 1]))

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

        print(x.size())
        x = x.view(bs, 2048, 2, 2)

        for i in range(len(self.tconvs)):
            x = self.tconvs[i](x)
            if i < len(self.tconvs) - 1:
                x = self.lrelu(x)

        return x


if __name__ == '__main__':
    device = torch.device("cpu:0")
    model = ConvAutoencoderLumaRelu12([1,4, 8, 32], [8092, 8092]).to(device).eval()
    x = torch.randn(1, 2, 64, 64, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
