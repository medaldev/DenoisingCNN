import datetime
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import torch

import common.fstream
import datageneration.generators
import models
from applications import model_manager
from common.fstream import (create_dir_of_file_if_not_exists, create_dir_if_not_exists)
from dataloaders import FeatureLoader2d
from testing.basic_test import basic_test
from datetime import datetime
import time
from common.stream import printProgressBar

import itertools

from .abstract_environment import AbstractEnvironment


class PolyFeaturesEnv(AbstractEnvironment):

    def __init__(self, name_model, name_dataset, path_base, device_name="cuda"):
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.path_base = path_base
        self.model = None
        self.device_name = device_name
        self.device = torch.device(device_name)

        self.path_save_train_plots = os.path.join(path_base, "runs", "train_plots", name_dataset,
                                                  name_model)

        self.path_train = os.path.join(path_base, "data", "datasets", name_dataset, "train", "calculations")
        self.path_test = os.path.join(path_base, "data", "datasets", name_dataset, "val", "calculations")

        self.train_features_loaders = []
        self.val_features_loaders = []

        self.val_target_loader = None
        self.train_target_loader = None

        self.train_batch_size = None
        self.val_batch_size = None

        self.train_count = None
        self.val_count = None

        self.train_losses, self.test_losses = [], []

    def set_batch_size(self, train_batch_size, val_batch_size):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        return self

    def clear_features_and_targets(self):
        self.train_features_loaders.clear()
        self.val_features_loaders.clear()
        self.train_target_loader = None
        self.val_target_loader = None


    def load_feature(self, width, height, feature_name, mapper, transform=None):

        self.train_features_loaders.append(
            FeatureLoader2d(
                self.path_train, feature_name, self.device, self.train_batch_size, width, height, mapper, transform
            )
        )

        if self.train_count is None:
            self.train_count = len(self.train_features_loaders[-1])

        self.val_features_loaders.append(
            FeatureLoader2d(
                self.path_test, feature_name, self.device, self.val_batch_size, width, height, mapper, transform
            )
        )

        if self.val_count is None:
            self.val_count = len(self.val_features_loaders[-1])

        return self

    def set_target(self, width, height, target_name, mapper, transform=None):
        self.train_target_loader = \
            FeatureLoader2d(
                self.path_train, target_name, self.device, self.train_batch_size, width, height, mapper, transform
            )

        # print(self.train_count, len(self.train_target_loader))
        assert self.train_count == len(self.train_target_loader)

        self.val_target_loader = \
            FeatureLoader2d(
                self.path_test, target_name, self.device, self.val_batch_size, width, height, mapper, transform
            )


        assert self.val_count == len(self.val_target_loader)

        return self

    def plot_batch(self, concrete):
        fig, axes = plt.subplots(self.val_batch_size, 2, figsize=(10, self.val_batch_size * 2))

        data_features = [fl[concrete] for fl in self.val_features_loaders]
        data_target = self.val_target_loader[concrete].resize(self.val_batch_size, self.val_target_loader.height, self.val_target_loader.width).detach().tolist()

        with torch.no_grad():
            outputs = self.model(*data_features).resize(self.val_batch_size, self.val_target_loader.height, self.val_target_loader.width).detach().tolist()

        images = []
        for k in range(self.val_batch_size):

            images.append(axes[k, 0].imshow(data_target[k], cmap="jet"))
            images.append(axes[k, 1].imshow(outputs[k], cmap="jet"))

        for im in images:
            fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, format='%.7f')

        plt.tight_layout()

        plt.xticks([]), plt.yticks([])

        plt.show()

        plt.clf()
        matplotlib.pyplot.close()


    def train(self, epochs, criterion=None, optimizer=None, scheduler=None, step_saving=True, step_plotting=True):

        # Loss function
        if not criterion:
            criterion = get_loss()

        # Optimizer
        if not optimizer:
            optimizer = get_optimizer(self.model)


        for epoch in range(epochs):

            # monitor training loss
            train_loss = 0.0
            test_loss = 0.0

            itr, itr_test = 0, 0
            start_point = time.time()
            # Training
            for row in zip(*self.train_features_loaders + [self.train_target_loader]):
                data_features = list(row)
                data_target = data_features.pop()

                optimizer.zero_grad()
                outputs = self.model(*data_features)
                loss = criterion(outputs, data_target)
                loss.backward()
                optimizer.step()

                loss_deltha = loss.item()
                train_loss += loss_deltha / self.train_count
                itr += 1
                # print("-iteration", itr, "deltha_loss", loss_deltha/len(train_noisy_loader))
                printProgressBar(itr, self.train_count, prefix='Training progress:', suffix='Complete',
                                 length=50)
            print()

            with torch.no_grad():

                for row in zip(*self.val_features_loaders + [self.val_target_loader]):
                    data_features = list(row)
                    data_target = data_features.pop()
                    outputs = self.model(*data_features)
                    loss = criterion(outputs, data_target)

                    loss_deltha = loss.item()
                    test_loss += loss_deltha / self.val_count
                    itr_test += 1
                    printProgressBar(itr_test, self.val_count, prefix='Validating progress:', suffix='Complete',
                                     length=50)
                print()

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidating Loss: {:.6f} \tTime: {:.2f} m'.
                  format(epoch + 1, train_loss,test_loss, (time.time() - start_point) / 60,2))

            if step_saving:
                if len(self.test_losses) and test_loss < min(self.test_losses):
                    self.save(onnx=False, pth=False)

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            if scheduler:
                scheduler.step()


            print("\n", "=" * 100, "\n", sep="")




def get_loss():
    return torch.nn.MSELoss()

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)



if __name__ == '__main__':
    pass
