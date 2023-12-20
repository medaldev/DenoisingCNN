import numpy as np
import matplotlib.pyplot as plt


class Figure:
    def draw(self, x, y) -> float:
        pass


class Point(Figure):
    def __init__(self, x0, y0):
        self.x0 = x0
        self.y0 = y0

    def draw(self, x, y) -> float:
        if x == self.x0 and y == self.y0:
            return 1.0
        return 0.0


class Circle(Figure):
    def __init__(self, x0, y0, r):
        self.x0 = x0
        self.y0 = y0
        self.r = r

    def draw(self, x, y) -> float:
        if (x - self.x0) ** 2 + (y - self.y0) ** 2 < self.r ** 2:
            return 1.0
        return 0.0


class Rectangle(Figure):
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height

    def draw(self, x, y) -> float:
        if self.x0 - self.width / 2 < x < self.x0 + self.width / 2 and self.y0 - self.height / 2 < y < self.y0 + self.height / 2:
            return 1.0
        return 0.0


class Triangle(Figure):
    def __init__(self, x1, y1, x2, y2, x3, y3):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.x3, self.y3 = x3, y3

    @staticmethod
    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    def draw(self, x, y) -> float:
        d1 = self.sign(x, y, self.x1, self.y1, self.x2, self.y2)
        d2 = self.sign(x, y, self.x2, self.y2, self.x3, self.y3)
        d3 = self.sign(x, y, self.x3, self.y3, self.x1, self.y1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        if not (has_neg and has_pos):
            return 1.0
        return 0.0


class Polygon(Figure):
    def __init__(self, points: list[tuple[float, float]]):
        self.points = points

    def is_point_in_polygon(self, x: float, y: float) -> bool:
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for point in self.points:
            min_x = min(point[0], min_x)
            max_x = max(point[0], max_x)
            min_y = min(point[1], min_y)
            max_y = max(point[1], max_y)

        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False

        inside = False
        j = len(self.points) - 1
        for i in range(len(self.points)):
            if ((self.points[i][1] > y) != (self.points[j][1] > y)) and \
                    (x < (self.points[j][0] - self.points[i][0]) * (y - self.points[i][1]) / (self.points[j][1] - self.points[i][1]) +
                     self.points[i][0]):
                inside = not inside
            j = i

        return inside

    def draw(self, x, y) -> float:
        if self.is_point_in_polygon(x, y):
            return 1.0
        return 0.0



def gen_matrix(n: int, m: int, cell_size: int, figures: list[Figure]) -> np.ndarray:
    height, width = n * cell_size, m * cell_size
    matrix = np.zeros((height, width))
    for i in range(height):
        y = i // cell_size
        for j in range(width):
            x = j // cell_size
            for fig in figures:
                matrix[i][j] = max(fig.draw(x, y), matrix[i][j])
    return matrix


def print_matrix(array: np.ndarray) -> None:
    for row in array:
        print(*list(row))


if __name__ == '__main__':
    arr = gen_matrix(300, 300, 1, [
        Circle(90, 90, 50),
        Point(20, 20),
        Rectangle(200, 200, 25, 50),
        Triangle(25, 200, 100, 230, 70, 260),
        Polygon([(170, 25), (180, 130), (200, 55), (250, 75), (220, 30), (240, 20)]),
        Polygon([(10, 25), (20, 50), (30, 25), (40, 50), (50, 25), (60, 50), (70, 25)])
    ])

    print_matrix(arr)

    plt.imshow(arr)
    plt.savefig("./res.png")
