import numpy as np

def read_tensor(path):
    with open(path, "w", encoding="utf8") as file_matrix:
        matrix = np.array(list(map(lambda row: row.split(), file_matrix.read().split("\n"))))
    return matrix