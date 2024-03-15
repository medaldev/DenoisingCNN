import torch
from torch import nn


class K_abs_Uvych_k0_v2(nn.Module):
    def __init__(self, size_uvych_parts: int, size_k_res: int):
        super(K_abs_Uvych_k0_v2, self).__init__()

        # Encoder
        self.size_uvych_parts: int = size_uvych_parts
        self.size_k_res: int = size_k_res

        self.linear_block = nn.Sequential(
            nn.Linear(2 * size_uvych_parts + 1, 2 * size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(2 * size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_uvych_parts),
            nn.LeakyReLU(),
            nn.Linear(size_uvych_parts, size_k_res),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()

        self.dp = nn.Dropout()

    def forward(self, x1, x2, k0):
        bs = x1.size()[0]
        x = torch.cat((x1, x2)).view(2 * self.size_uvych_parts, bs)
        x = torch.cat((x, k0.view(1, bs))).view(bs, 2 * self.size_uvych_parts + 1)
        x = self.linear_block(x).view(bs, 1, self.size_k_res)

        return x


def test():
    device = torch.device("cpu:0")
    model = K_abs_Uvych_k0_v2(900, 900).to(device).eval()
    x1 = torch.randn(1, 1, 900, device=device)
    x2 = torch.randn(1, 1, 900, device=device)
    k0 = torch.randn(1, 1, 1, 1, device=device)
    print(model(x1, x2, k0).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))


test()