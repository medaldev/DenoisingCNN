from datageneration.components.figure import Figure


class Point(Figure):
    def __init__(self, x0, y0):
        self.x0 = x0
        self.y0 = y0

    def draw(self, x, y) -> float:
        if x == self.x0 and y == self.y0:
            return 1.0
        return 0.0