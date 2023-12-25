import random

import numpy as np

from datagenerators.components.figure import Figure


class Surface:
    def __init__(self, height: int, width: int, cell_size: int):
        self.height = height
        self.width = width
        self.cell_size = cell_size

        self.figures = []

    def add_figure(self, figure: Figure):
        self.figures.append(figure)
        return self

    def matrix(self):
        matrix = np.zeros((self.height, self.width))
        for i in range(self.height):
            y = i // self.cell_size
            for j in range(self.width):
                x = j // self.cell_size
                for fig in self.figures:
                    matrix[i][j] = max(fig.draw(x, y), matrix[i][j])
        return matrix


def add_noise(matrix: np.ndarray, cell_size) -> np.ndarray:
    height, width = matrix.shape
    for I in range(0, height, cell_size):
        for J in range(0, width, cell_size):
            noise = random.random()
            for i in range(I, I + cell_size):
                for j in range(J, J + cell_size):
                    matrix[i][j] += noise
    return matrix


def print_matrix(array: np.ndarray) -> None:
    for row in array:
        print(*list(row))
