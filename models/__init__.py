from models.autoencoders.ConvAutoEncoder import ConvAutoencoder
from models.autoencoders.ConvAutoencoderLuma import ConvAutoencoderLuma


def get_basic_model():
    return ConvAutoencoderLuma()
