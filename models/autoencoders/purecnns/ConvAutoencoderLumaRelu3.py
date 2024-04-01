from torch import nn
import torch


class ConvAutoencoderLumaRelu3(nn.Module):
    def __init__(self):
        super(ConvAutoencoderLumaRelu3, self).__init__()

        # Encoder
        self.ln1 = nn.Linear(900, 1024)
        self.ln2 = nn.Linear(1024, 2048)
        self.ln3 = nn.Linear(2048, 4096)
        self.ln4 = nn.Linear(4096, 8192)
        self.ln5 = nn.Linear(8192, 8192)
        self.ln6 = nn.Linear(8192, 8192)

        #self.lin1 = nn.Linear

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=2048, kernel_size=2, stride=2, padding=0)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=2048, out_channels=2048, kernel_size=2, stride=2, padding=0)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=2048, out_channels=2048, kernel_size=2, stride=2, padding=0)
        self.t_conv4 = nn.ConvTranspose2d(in_channels=2048, out_channels=2048, kernel_size=2, stride=1, padding=0)
        self.t_conv5 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=1, padding=0)
        self.t_conv6 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=1, padding=0)
        self.t_conv7 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=1, padding=0)
        self.t_conv8 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=1, padding=0)
        self.t_conv9 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.t_conv10 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.t_conv11 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.t_conv12 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.t_conv13 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0)
        self.t_conv14 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.bn2 = nn.BatchNorm1d(num_features=2048)
        self.bn3 = nn.BatchNorm1d(num_features=4096)
        self.bn4 = nn.BatchNorm1d(num_features=8192)

        self.bn2d = nn.BatchNorm2d(100)

        self.dp = nn.Dropout()



    def forward(self, x):
        bs = x.size()[0]

        x = x.view(x.size()[0], 900)

        x = self.ln1(x)
        x = self.bn1(x)
        x1 = self.selu(x)

        x = self.ln2(x1)
        x = self.bn2(x)
        x2 = self.selu(x)

        x = self.ln3(x2)
        x3 = self.selu(x)

        x = self.ln4(x3)
        x = self.bn4(x)
        x4 = self.selu(x)

        x = self.ln5(x4)
        x5 = self.selu(x)

        x = self.ln6(x5)
        x = self.bn4(x)
        x = self.selu(x)


        x = x.view(bs, 2048, 2, 2)


        x = self.t_conv1(x)
        x = self.selu(x)

        x = self.t_conv2(x)
        x = self.selu(x)

        x = self.t_conv3(x)
        x = self.selu(x)

        x = self.t_conv4(x)
        x = self.selu(x)

        x = self.t_conv5(x)
        x = self.selu(x)

        x = self.t_conv6(x)
        x = self.selu(x)

        x = self.t_conv7(x)
        x = self.selu(x)

        x = self.t_conv8(x)
        x = self.selu(x)

        x = self.t_conv9(x)
        x = self.selu(x)

        x = self.t_conv10(x)
        x = self.selu(x)

        x = self.t_conv11(x)
        x = self.selu(x)

        x = self.t_conv12(x)
        x = self.selu(x)

        x = self.t_conv13(x)
        x = self.selu(x)

        x = self.t_conv14(x)

        return x


if __name__ == '__main__':
    model = ConvAutoencoderLumaRelu3().eval()
    x = torch.randn(1, 1, 30, 30)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
