import os
import random

from torch.utils.data import Dataset
from torch.nn.init import *

from torchvision import transforms

import numpy as np

from random import shuffle
from common.stream import printProgressBar

import time


class FeatureLoader2d(Dataset):

    def __init__(self, data_dir, feature, device, batch_size, shape, read_tensor, transform=None, lazy_load=False, pct_load=None, dtype=None):

        if dtype is None:
            dtype = torch.float

        self.dtype = dtype
        self.feature = feature
        self.shape = shape

        self.read_tensor = read_tensor

        self.all_files = [os.path.join(data_dir, d, feature) for d in os.listdir(data_dir) if
                          os.path.isfile(os.path.join(data_dir, d, feature))]
        if pct_load:
            self.all_files = self.all_files[:int(len(self.all_files) * pct_load)]
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
        self.lazy_load = lazy_load

        if not lazy_load:
            self.items = []
            for idx in range(len(self.data)):
                self.items.append(self.load_item(idx))
                printProgressBar(idx, len(self.data), prefix='Data loading progress:', suffix='Complete',
                                 length=50)


    def shuffle(self):
        shuffle(self.data)

    def get_rand_single(self):
        return self.read_tensor(self.data[random.randint(0, len(self.data) - 1)])

    def __len__(self):
        return len(self.data)

    def load_item(self, idx):
        content = list(map(self.read_tensor, self.data[idx]))

        batch = torch.tensor(np.array(list(map(lambda el: np.array(el),
                                               content))), dtype=self.dtype, device=self.device)

        if self.transform:
            batch = self.transform(batch)

        batch = batch.view(
            self.batch_size, *self.shape)

        return batch

    def __getitem__(self, idx):
        if self.lazy_load:
            return self.load_item(idx)
        else:
            return self.items[idx]
