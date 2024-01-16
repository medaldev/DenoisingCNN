from models.autoencoders.ConvAutoEncoder import ConvAutoencoder
from models.autoencoders.ConvAutoencoderLuma import ConvAutoencoderLuma
from models.autoencoders.ConvAutoencoderLumaRelu import ConvAutoencoderLumaRelu
from models.autoencoders.ConvAutoencoderLumaRelu2 import ConvAutoencoderLumaRelu2

from .experimental import *

def get_basic_model():
    return ConvAutoencoderLuma()
