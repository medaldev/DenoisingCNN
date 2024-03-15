import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn

from models.reconstructors.reconnet import ReconNet


# Create the feature extractor

class ConvAutoencoderLumaRelu6(torch.nn.Module):
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


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = ConvAutoencoderLumaRelu6().to(device).eval()
    x = torch.randn(1, 1, 30, 30, device=device)
    print(model(x).size())

    print("params", sum(p.numel() for p in model.parameters() if p.requires_grad))
