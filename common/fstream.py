import numpy as np
import os

from common.matrix import rescale_array
from common.value import rescale_val


def read_matrix(path):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(
            list(map(lambda row: list(map(lambda el: float(el), row.split())), file_matrix.read().strip().split("\n"))))
    return matrix


def read_tensor(path):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = rescale_array(np.array(list(
            map(lambda row: list(map(lambda el: float(el), row.split())), file_matrix.read().strip().split("\n")))))
    return matrix


def array_save(matrix: np.ndarray, save_path: str):
    h, w = matrix.shape
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf8") as file:
        for i in range(h):
            for j in range(w):
                print(matrix[i][j], file=file, end=" ")
            print(file=file)


def arrayToCsv(matrix: np.ndarray, save_path: str, with_rescale_args=True):
    h, w = matrix.shape
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf8") as csv_file:
        for i in range(h):
            for j in range(w):
                if with_rescale_args:
                    print(rescale_val(i, 0, h), rescale_val(j, 0, w), matrix[i][j], file=csv_file, end=", ")
                else:
                    print(i, j, matrix[i][j], file=csv_file, end=", ")
