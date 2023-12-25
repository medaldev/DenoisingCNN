from datagenerators.components.figure import Figure


class Circle(Figure):
    def __init__(self, x0, y0, r):
        self.x0 = x0
        self.y0 = y0
        self.r = r

    def draw(self, x, y) -> float:
        if (x - self.x0) ** 2 + (y - self.y0) ** 2 < self.r ** 2:
            return 1.0
        return 0.0
