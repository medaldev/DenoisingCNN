import random


class GenK0Const:

    def __init__(self, val):
        self.val = val

    def __iter__(self):
        return self

    def __next__(self):
        return self.val


class GenK0InRange:

    def __init__(self, min_val=3, max_val=100):
        self.min_val = min_val
        self.max_val = max_val

    def __iter__(self):
        return self

    def __next__(self):
        return random.randint(self.min_val, self.max_val)
