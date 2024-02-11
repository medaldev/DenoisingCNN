import torchvision
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
import torch.nn.functional as F
import torch

class ConvTwoToOne2(nn.Module):
    def __init__(self, final_shape=(80, 80)):
        super(ConvTwoToOne2, self).__init__()

        super().__init__()

        # Load a pretrained ResNet model
        backbone = torchvision.models.resnet50(pretrained=True)

        for param in backbone.parameters():
            param.requires_grad = True

        self.pre_extract = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=1)
        self.encoder = create_feature_extractor(backbone, return_nodes={'layer4': 'features'})
        # Define your decoder to match the output feature size of 'features'
        self.decoder = nn.Sequential(
            nn.Linear(24576, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 80 * 80),
        )

        self.fl = nn.Flatten()

    def forward(self, x1, x2):
        bs = x1.size()[0]

        x1 = self.encoder(self.pre_extract(x1))['features']
        x2 = self.encoder(self.pre_extract(x2))['features']

        x = torch.cat([x1, x2], dim=1)
        x = self.fl(x)
        x = self.decoder(x)
        x = x.view(bs, 1, 80, 80)

        return x


def test_model():
    device = torch.device('cpu')
    model = ConvTwoToOne2().to(device)
    inp1 = torch.randn((4, 1, 80, 40), device=device)
    inp2 = torch.randn((4, 1, 80, 40), device=device)

    with torch.no_grad():
        print(model(inp1, inp2).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))

# test_model()

