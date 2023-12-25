import numpy as np
import random


def rescale_array(a, to=(0, +1)):
    return np.interp(a, (a.min(), a.max()), to)


def add_noise(matrix: np.ndarray, cell_size) -> np.ndarray:
    height, width = matrix.shape
    for I in range(0, height, cell_size):
        for J in range(0, width, cell_size):
            noise = random.random()
            for i in range(I, I + cell_size):
                for j in range(J, J + cell_size):
                    matrix[i][j] += noise
    return matrix
