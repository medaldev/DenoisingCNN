import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class Chb(nn.Module):
    def __init__(self, in_dim: int, k: int):
        super().__init__()
        self.in_dim = in_dim
        self.k = k
        self.cn = nn.ParameterList([nn.Parameter(torch.randn(in_dim, )) for _ in range(k)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.zeros_like(x)

        for i in range(self.k):
            res += self.cn[i] * self.pn(x, i)
        return res

    def pn(self, x: torch.Tensor, n: int) -> torch.Tensor:
        match n:
            case 0:
                return torch.ones_like(x)
            case 1:
                return 1.0 * x
            case 2:
                return 0.5 * (3 * torch.pow(x, 2) - 1)
            case 3:
                return 0.5 * (5 * torch.pow(x, 3) - 3 * x)
            case 4:
                return 1./8. * (35 * torch.pow(x, 4) - 30 * torch.pow(x, 2) + 3)
            case 5:
                return 1./8. * (63 * torch.pow(x, 5) - 70. * torch.pow(x, 3) + 15. * x)
            case _:
                raise Exception




if __name__ == '__main__':
    device = torch.device("cpu")
    model = Chb(1, 6)
    test_x = torch.randn(32, 10)

    # with torch.no_grad():
    #     print(model(test_x).size())

    lr = 0.01
    betas = (0.9, 0.999)
    history = []

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-8)


    my_x = np.arange(0., 1., 20 / 1000)
    my_y = my_x ** 2


    tensor_x = torch.Tensor(my_x)  # transform to torch tensor
    tensor_y = torch.Tensor(my_y)
    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = DataLoader(my_dataset, batch_size=32)  # create your dataloader

    epochs = 40
    for epoch in tqdm(range(epochs)):
        for data_x, data_y in dataloader:
            optimizer.zero_grad()
            pred = model(data_x)
            loss = criterion(pred, data_y)
            loss.backward()
            optimizer.step()

            history.append(loss.detach().numpy())
            print("Loss: ", history[-1])

    import matplotlib.pyplot as plt
    plt.plot(history)
    plt.show()

    plt.clf()
    plt.plot(my_x, my_y)
    plt.plot(my_x,  model(tensor_x).detach().numpy())
    plt.show()