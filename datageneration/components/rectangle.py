from datageneration.components.figure import Figure


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
