import random


class GenPctNoiseConst:

    def __init__(self, val):
        self.val = val

    def __iter__(self):
        return self

    def __next__(self):
        return self.val


class GenPctUniformNoiseRange:

    def __init__(self, a=0.01, b=1.0):
        self.a = a
        self.b = b

    def __iter__(self):
        return self

    def __next__(self):
        return random.uniform(self.a, self.b)


class GenPctUniformNoiseSet:

    def __init__(self, values=None):
        if values is None:
            values = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        self.values = values

    def __iter__(self):
        return self

    def __next__(self):
        return random.choice(self.values)