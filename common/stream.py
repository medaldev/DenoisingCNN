import numpy as np


def print_matrix(array: np.ndarray) -> None:
    for row in array:
        print(*list(row))
