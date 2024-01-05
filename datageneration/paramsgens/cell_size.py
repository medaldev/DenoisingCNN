import random


class GenCellSizeConst:

    def __init__(self, val=2):
        self.val = val

    def __iter__(self):
        return self

    def __next__(self):
        return self.val


class GenCellSizeInSet:

    def __init__(self, values=None):
        if values is None:
            values = [1, 2]
        self.values = values

    def __iter__(self):
        return self

    def __next__(self):
        return random.choice(self.values)