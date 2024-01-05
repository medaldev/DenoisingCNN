from models.autoencoders.ConvAutoEncoder import ConvAutoencoder
from models.autoencoders.ConvAutoencoderLuma import ConvAutoencoderLuma
from models.autoencoders.ConvAutoencoderLumaRelu import ConvAutoencoderLumaRelu


def get_basic_model():
    return ConvAutoencoderLuma()
