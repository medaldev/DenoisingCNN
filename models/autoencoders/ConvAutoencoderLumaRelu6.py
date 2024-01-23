from torch import nn
import torch

import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
import torch.nn.functional as F



# Create the feature extractor

class ImageAutoencoder(torch.nn.Module):
    def __init__(self, final_shape=(30, 30)):
        super().__init__()

        # Load a pretrained ResNet model
        backbone = models.resnet50(pretrained=True)

        for param in backbone.parameters():
            param.requires_grad = False

        return_nodes = {
            'layer4': 'features',
        }

        self.pre_extract = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=1)
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)
        # Define your decoder to match the output feature size of 'features'
        self.decoder = ReconNet(vector_len=2048, final_shape=final_shape)
        self.fl = nn.Flatten()

    def forward(self, x):
        # Extract features
        x = self.pre_extract(x)
        encoded_features = self.encoder(x)['features']
        # Reconstruct the image
        reconstructed = self.decoder(self.fl(encoded_features))
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
    model = ImageAutoencoder().to(device).eval()
    x = torch.randn(1, 1, 30, 30, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
