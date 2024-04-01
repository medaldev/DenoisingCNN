from torch import nn


class NoiseMatrixPredictor_v2(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, hidden_size ):
        super(NoiseMatrixPredictor_v2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.width = width
        self.height = height

        self.orig_size = width * height
        self.hidden_size = hidden_size


        self.relu = nn.LeakyReLU()

        self.ft = nn.Sequential(
            nn.Linear(self.orig_size, self.orig_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, self.orig_size)
        )


    def forward(self, x):
        bs = x.size()[0]
        # ========== encoder ==========
        x = self.conv0(x)
        x = self.relu(x)
        res0 = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        res1 = x
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        res2 = x
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        res3 = x
        x = self.conv7(x)
        x = self.relu(x)
        # ========== decoder ==========
        x = self.conv_transpose0(x)
        x = self.relu(x)
        x += res3
        x = self.conv_transpose1(x)
        x = self.relu(x)
        x = self.conv_transpose2(x)
        x = self.relu(x)
        x += res2
        x = self.conv_transpose3(x)
        x = self.relu(x)
        x = self.conv_transpose4(x)
        x = self.relu(x)
        x += res1
        x = self.conv_transpose5(x)
        x = self.relu(x)
        x = self.conv_transpose6(x)
        x = self.relu(x)
        x += res0
        x = self.conv_transpose7(x)
        x = self.relu(x)
        x = x.view(bs, self.height * self.width)
        x = self.ft(x)
        x = x.view(bs, self.out_channels, self.height, self.width)
        return x


import torch

if __name__ == '__main__':
    device = torch.device("cpu:0")
    model = NoiseMatrixPredictor_v1(1, 1, 80, 80, 80 * 80).to(device).eval()
    x = torch.randn(4, 1, 80, 80, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
