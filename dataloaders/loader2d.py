import os

from torch.utils.data import Dataset, DataLoader
from torch.nn.init import *

from torchvision import transforms, utils, datasets, models

import numpy as np

from .functions import read_tensor


class SimpleLoader2d(Dataset):

    def __init__(self, data_dir, device, batch_size, transform=None):

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        originals = [os.path.isfile(os.path.join(data_dir, "original", f))
                     for f in os.listdir(os.path.join(data_dir, "original"))
                     if os.path.isfile(os.path.join(data_dir, "original", f))]

        noised = [os.path.isfile(os.path.join(data_dir, "noised", f))
                  for f in os.listdir(os.path.join(data_dir, "noised"))
                  if os.path.isfile(os.path.join(data_dir, "noised", f))]

        self.data = []

        assert len(originals) == len(noised)

        self.count_batches = len(originals) // batch_size

        for i in range(self.count_batches):

            batch = []
            for _ in range(batch_size):
                (x, y) = (noised.pop(), originals.pop())

                assert x == y

                batch.append((x, y))

            self.data.append(batch)

        self.transform = transform

        self.batch_size = batch_size

        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        batch = list(map(lambda el: torch.tensor(np.array(el), dtype=torch.float, device=self.device),
                         map(read_tensor, self.data[idx])))

        return batch
