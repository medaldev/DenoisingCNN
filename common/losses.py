import torch


class MseCoeffLoss(torch.nn.MSELoss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', coeff: float = 1.0) -> None:
        super().__init__(size_average, reduce, reduction)
        self.coeff = coeff

    def forward(self, input, target):
        return self.coeff * super().forward(input, target)


class L1CoeffLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', coeff: float = 1.0) -> None:
        super().__init__(size_average, reduce, reduction)
        self.coeff = coeff

    def forward(self, input, target):
        return self.coeff * super().forward(input, target)
