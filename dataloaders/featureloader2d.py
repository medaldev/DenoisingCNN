import os
import random

from torch.utils.data import Dataset
from torch.nn.init import *

from torchvision import transforms

import numpy as np

from random import shuffle


class FeatureLoader2d(Dataset):

    def __init__(self, data_dir, feature, device, batch_size, width, height, read_tensor, transform=None):

        self.feature = feature
        self.width = width
        self.height = height

        self.read_tensor = read_tensor

        self.all_files = [os.path.join(data_dir, d, feature) for d in os.listdir(data_dir) if
                          os.path.isfile(os.path.join(data_dir, d, feature))]
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

    def get_rand_single(self):
        return self.read_tensor(self.data[random.randint(0, len(self.data) - 1)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = list(map(self.read_tensor, self.data[idx]))

        batch = torch.tensor(np.array(list(map(lambda el: np.array(el),
                                               content))), dtype=torch.float, device=self.device)

        if self.transform:
            batch = self.transform(batch)

        batch = batch.resize(
            self.batch_size, 1, self.height, self.width)

        return batch
