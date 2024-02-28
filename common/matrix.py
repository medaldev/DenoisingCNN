import numpy as np
import random

import torch


def rescale_array(a, to=(0, +1)):
    return np.interp(a, (a.min(), a.max()), to)


def rescale_tensor(outmap):
    outmap_min = torch.min(outmap)
    outmap_max = torch.max(outmap)

    return (outmap - outmap_min) / (outmap_max - outmap_min)  # Broadcasting rules apply


def read_tensor_and_norm(path, norm_coeff):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(list(map(lambda row: list(map(lambda el: float(el) / norm_coeff, row.split())),
                                   file_matrix.read().strip().split("\n"))))
    return matrix


def add_noise(matrix: np.ndarray, cell_size, pct=0.1) -> np.ndarray:
    height, width = matrix.shape
    noise_min, noise_max = 0.0, np.abs(np.max(matrix) - np.min(matrix)) * pct
    for I in range(0, height, cell_size):
        for J in range(0, width, cell_size):
            noise = random.uniform(noise_min, noise_max)
            for i in range(I, I + cell_size):
                for j in range(J, J + cell_size):
                    matrix[i][j] += noise
    return matrix

if __name__ == '__main__':
    with torch.no_grad():
        a = torch.randn((3, 4), dtype=torch.float64)
        b = a * 1e3

        c = rescale_tensor(a)
        d = rescale_tensor(b)

        print(a)
        print(b)
        print(c)
        print(d)
