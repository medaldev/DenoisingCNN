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



