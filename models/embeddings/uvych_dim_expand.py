from torch import nn


class UvychDimExpand(nn.Module):
    def __init__(self, width, height, hidden_size):
        super(UvychDimExpand, self).__init__()

        self.width = width
        self.height = height
        self.orig_size = width * height
        self.hidden_size = hidden_size

        self.ft = nn.Sequential(
            nn.Linear(self.orig_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        self.inv = nn.Linear(hidden_size, self.orig_size)

    def forward(self, x):
        bs = x.size()[0]
        orig_shape = tuple(x.shape)
        x = x.view(bs, self.height * self.width)
        h = self.ft(x)
        x = self.inv(h)
        x = x.view(orig_shape)

        return x

    def embedding(self, x):
        h = self.ft(x)
        return h


import torch

if __name__ == '__main__':
    device = torch.device("cpu:0")
    model = UvychDimExpand(80, 80, 80 * 80 * 2).to(device).eval()
    x = torch.randn(4, 1,  80,  80, device=device)
    print()
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
