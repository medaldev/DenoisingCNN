from models.autoencoders.ConvAutoEncoder import ConvAutoencoder
from models.autoencoders.ConvAutoencoderLuma import ConvAutoencoderLuma
from models.autoencoders.ConvAutoencoderLumaRelu import ConvAutoencoderLumaRelu
from models.autoencoders.ConvAutoencoderLumaRelu2 import ConvAutoencoderLumaRelu2
from models.autoencoders.ConvAutoencoderLumaRelu3 import ConvAutoencoderLumaRelu3
from models.autoencoders.ConvAutoencoderLumaRelu4 import ConvAutoencoderLumaRelu4
from models.autoencoders.ConvAutoencoderLumaRelu5 import ConvAutoencoderLumaRelu5
from models.autoencoders.ConvAutoencoderLumaRelu6 import ImageAutoencoder
from models.autoencoders.ConvAutoencoderLumaRelu7 import ImageAutoencoder2

from . import reconstructors
from . import predictors

from models.autoencoders import UnetAutoencoder
from . import poly_features

from .experimental import *

import models.diffusion_models

from .vqgan_custom import vqgan
def get_basic_model():
    return ConvAutoencoderLuma()


import torch.nn.functional as F

from torch import nn

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

