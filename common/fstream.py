import numpy as np
import os
import json
import ast

from common.matrix import rescale_array
from common.value import rescale_val


def read_vector_min_max(path) -> np.ndarray:
    vector = read_as_vector(path)
    return np.array([np.min(vector), np.max(vector)])


def read_matrix(path, coeff_norm=1.0):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(
            list(map(lambda row: list(map(lambda el: float(el), row.split())),
                     file_matrix.read().strip().split("\n")))) / coeff_norm
    return matrix


def read_as_vector(path):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(list(map(lambda el: float(el), file_matrix.read().split())))
    return matrix


def read_mc_tensor(path, rescale_each=None):
    with open(path, "r", encoding="utf8") as file_tensor:
        data = json.load(file_tensor)
    if rescale_each:
        for i in range(len(data)):
            data[i] = rescale_array(np.array(data[i]), rescale_each[1], rescale_each[0])
    return np.array(data)


def read_tensor(path, rescale=True):
    if not rescale:
        return read_matrix(path)
    matrix = rescale_array(read_matrix(path))
    return matrix


def array_save(matrix: np.ndarray, save_path: str):
    h, w = matrix.shape
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf8") as file:
        for i in range(h):
            for j in range(w):
                print(matrix[i][j], file=file, end="\t")
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


def create_dir_of_file_if_not_exists(path_to_file):
    if not os.path.exists(os.path.dirname(path_to_file)):
        os.makedirs(os.path.dirname(path_to_file))


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
