import os

from torch.utils.data import Dataset
from torch.nn.init import *

from torchvision import transforms

import numpy as np

from random import shuffle


class SimpleLoader2d(Dataset):

    def __init__(self, data_dir, device, batch_size, width, height, read_tensor, transform=None):

        self.width = width
        self.height = height

        self.read_tensor = read_tensor

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
                          os.path.isfile(os.path.join(data_dir, f))]
        # shuffle(self.all_files)
        self.data = []

        self.count_batches = len(self.all_files) // batch_size

        for i in range(self.count_batches):

            batch = []
            for _ in range(batch_size):
                batch.append(self.all_files.pop())

            self.data.append(batch)

        self.transform = transform

        self.batch_size = batch_size

        self.device = device

    def shuffle(self):
        shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = list(map(self.read_tensor, self.data[idx]))

        batch = torch.tensor(np.array(list(map(lambda el: np.array(el),
                                               content))), dtype=torch.float, device=self.device).resize(
            self.batch_size, 1, self.width, self.height)

        return batch, True
