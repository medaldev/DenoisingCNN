import numpy as np
from PIL import Image


def arrayToImage(matrix: np.ndarray, save_path="./res.png"):
    im = Image.fromarray(matrix).convert("RGB")
    im.save(save_path)
