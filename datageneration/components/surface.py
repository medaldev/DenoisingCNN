import random

import numpy as np

from datageneration.components.figure import Figure


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