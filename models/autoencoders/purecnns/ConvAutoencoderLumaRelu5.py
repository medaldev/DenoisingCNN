from torch import nn
import torch

class ConvAutoencoderLumaRelu5(nn.Module):
    def __init__(self):
        super(ConvAutoencoderLumaRelu5, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, stride=1, padding=0)
        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=1, padding=0)
        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=1, padding=0)
        self.conv9 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0)
        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0)
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=2, stride=2, padding=0)

        self.flatten1 = nn.Flatten()

        self.ln1 = nn.Linear(8192, 8192)

        #self.lin1 = nn.Linear

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2, padding=0)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0)
        self.t_conv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=1, padding=0)
        self.t_conv5 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=1, padding=0)
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

        x1 = self.conv1(x)
        x1 = self.selu(x1)

        x2 = self.conv2(x1)
        x2 = self.selu(x2)

        x3 = self.conv3(x2)
        x3 = self.selu(x3)

        x4 = self.conv4(x3)
        x4 = self.selu(x4)

        x5 = self.conv5(x4)
        x5 = self.selu(x5)

        x6 = self.conv6(x5)
        x6 = self.selu(x6)

        x7 = self.dp(x6)

        x7 = self.conv7(x7)
        x7 = self.selu(x7)

        x8 = self.conv8(x7)
        x8 = self.selu(x8)

        x9 = self.conv9(x8)
        x9 = self.selu(x9)

        x10 = self.dp(x9)

        x10 = self.conv10(x10)
        x10 = self.selu(x10)

        x = self.conv11(x10)
        x = self.selu(x)


        x = self.flatten1(x)

        x = self.ln1(x)
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
    device = torch.device("cuda:0")
    model = ConvAutoencoderLumaRelu5().to(device).eval()
    x = torch.randn(1, 1, 30, 30, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
