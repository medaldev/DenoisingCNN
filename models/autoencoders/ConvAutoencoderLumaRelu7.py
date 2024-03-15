from torch import nn
import torch

import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
import torch.nn.functional as F



# Create the feature extractor

class ConvAutoencoderLumaRelu7(torch.nn.Module):
    def __init__(self, final_shape=(30, 30)):
        super().__init__()

        # Load a pretrained ResNet model
        backbone = models.resnet50(pretrained=True)

        for param in backbone.parameters():
            param.requires_grad = False

        return_nodes = {
            'layer4': 'features',
        }

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(900, 8096),
            nn.Sigmoid(),
        )
        # Define your decoder to match the output feature size of 'features'
        self.decoder = nn.Sequential(
            nn.Linear(8096, 900),

        )
        self.fl = nn.Flatten()

    def forward(self, x):

        # Extract features
        encoded_features = self.encoder(x)
        # Reconstruct the image
        reconstructed = self.decoder(encoded_features)
        reconstructed = reconstructed.view(reconstructed.size()[0], 1, 30, 30)
        return reconstructed


class ReconNet(nn.Module):
  def __init__(self, vector_len=2048, final_shape=(30, 30)):
    super(ReconNet,self).__init__()

    self.width, self.height = final_shape

    self.fc1 = nn.Linear(int(vector_len),self.width * self.height)
    nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
    self.conv1 = nn.Conv2d(1,64,11,1,padding=5)
    nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
    self.conv2 = nn.Conv2d(64,32,1,1,padding=0)
    nn.init.normal_(self.conv2.weight, mean=0, std=0.1)
    self.conv3 = nn.Conv2d(32,1,7,1,padding=3)
    nn.init.normal_(self.conv3.weight, mean=0, std=0.1)
    self.conv4 = nn.Conv2d(1,64,11,1,padding=5)
    nn.init.normal_(self.conv4.weight, mean=0, std=0.1)
    self.conv5 = nn.Conv2d(64,32,1,1,padding=0)
    nn.init.normal_(self.conv5.weight, mean=0, std=0.1)
    self.conv6 = nn.Conv2d(32,1,7,1,padding=3)
    nn.init.normal_(self.conv6.weight, mean=0, std=0.1)

  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = x.view(-1,self.height, self.width)
    x = x.unsqueeze(1)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = self.conv6(x)

    return x


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = ConvAutoencoderLumaRelu7().to(device).eval()
    x = torch.randn(1, 1, 30, 30, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
