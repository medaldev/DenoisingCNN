from datagenerators.components.figure import Figure


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