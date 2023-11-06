import numpy as np

def read_tensor(path):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(list(map(lambda row: list(map(lambda el: float(el) / 50., row.split())), file_matrix.read().strip().split("\n"))))
    return matrix