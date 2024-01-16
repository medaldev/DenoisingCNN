import random

import numpy as np

from datageneration.components.figure import Figure


class Surface:
    def __init__(self, height: int, width: int, cell_size: int, k0: float):
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.k0 = k0
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
                if matrix[i][j] == 0:
                    matrix[i][j] = self.k0
        return matrix

    def geometry(self):
        matrix = np.zeros((self.height, self.width))
        for i in range(self.height):
            y = i // self.cell_size
            for j in range(self.width):
                x = j // self.cell_size
                for fig in self.figures:
                    matrix[i][j] = 1.0 if max(fig.draw(x, y), matrix[i][j]) else 0.0
                if matrix[i][j] == 0:
                    matrix[i][j] = 0.0
        return matrix

    def set_k0(self, k0: float):
        self.k0 = k0
