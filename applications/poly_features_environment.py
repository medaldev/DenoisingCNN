import datetime
import gc
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import common.fstream
import datageneration.generators
import models
from applications import model_manager
from common.fstream import (create_dir_of_file_if_not_exists, create_dir_if_not_exists)
from dataloaders.featureloader2d_v2 import FeatureLoader2d
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

        # self.path_save_train_plots = os.path.join(path_base, "runs", "train_plots", name_dataset,
        #                                           name_model)

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

    def set_name_model(self, name_model):
        self.name_model = name_model

    def set_batch_size(self, train_batch_size, val_batch_size):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        return self

    def clear_features_and_targets(self):
        self.train_features_loaders.clear()
        self.val_features_loaders.clear()
        self.train_target_loader = None
        self.val_target_loader = None


    def load_feature(self, shape, feature_name, mapper, transform=None, lazy_load=False, pct_load=None, dtype=None, ignore_not_exists=False):

        self.train_features_loaders.append(
            FeatureLoader2d(
                self.path_train, feature_name, self.device, self.train_batch_size, shape, mapper, transform, lazy_load, pct_load, dtype, ignore_not_exists
            )
        )

        if self.train_count is None:
            self.train_count = len(self.train_features_loaders[-1])

        self.val_features_loaders.append(
            FeatureLoader2d(
                self.path_test, feature_name, self.device, self.val_batch_size, shape, mapper, transform, lazy_load, pct_load, dtype, ignore_not_exists
            )
        )

        if self.val_count is None:
            self.val_count = len(self.val_features_loaders[-1])

        return self

    def set_target(self, shape, target_name, mapper, transform=None, lazy_load=False, pct_load=None, dtype=None, ignore_not_exists=False):
        self.train_target_loader = \
            FeatureLoader2d(
                self.path_train, target_name, self.device, self.train_batch_size, shape, mapper, transform, lazy_load, pct_load, dtype, ignore_not_exists
            )

        # print(self.train_count, len(self.train_target_loader))
        assert self.train_count == len(self.train_target_loader)

        self.val_target_loader = \
            FeatureLoader2d(
                self.path_test, target_name, self.device, self.val_batch_size, shape, mapper, transform, lazy_load, pct_load, dtype, ignore_not_exists
            )


        assert self.val_count == len(self.val_target_loader)

        return self

    def plot_batch(self, concrete, figsize=None, format='%.7f', wspace=0.3, hspace=0.1, size=None, func_postprocess=None):
        if func_postprocess is None:
            func_postprocess = lambda x: x

        if size is None:
            size = self.val_target_loader.shape[1:]
        if figsize is None:
            figsize = (10, self.val_batch_size * 3)
        fig, axes = plt.subplots(self.val_batch_size, 3, figsize=figsize)

        data_features = [fl[concrete] for fl in self.val_features_loaders]
        data_target = self.val_target_loader[concrete].resize(self.val_batch_size, *size).cpu().detach().numpy()

        with torch.no_grad():
            outputs = self.model(*data_features).resize(self.val_batch_size, *size).cpu().detach().numpy()

        images = []

        axes[0, 0].set_title("Real", pad=20)
        axes[0, 1].set_title("Pred", pad=20)
        axes[0, 2].set_title("Diff", pad=20)
        for k in range(self.val_batch_size):

            images.append(axes[k, 0].imshow(func_postprocess(data_target[k]), cmap="jet"))
            images.append(axes[k, 1].imshow(func_postprocess(outputs[k]), cmap="jet"))
            images.append(axes[k, 2].imshow(np.abs(func_postprocess(outputs[k]) - func_postprocess(data_target[k])), cmap="jet"))

            print(data_target[k])
            print(outputs[k])

        for im in images:
            fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, format=format)

        plt.tight_layout()
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

        for row in axes:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.show()

        plt.clf()
        matplotlib.pyplot.close()

    def plot_batch_with_inputs(self, concrete, figsize=None, format='%.7f', wspace=0.3, hspace=0.1):
        if figsize is None:
            figsize = (10, self.val_batch_size * 2)
        fig, axes = plt.subplots(self.val_batch_size, 2 + len(self.val_features_loaders), figsize=figsize)

        data_features = [fl[concrete] for fl in self.val_features_loaders]
        data_target = self.val_target_loader[concrete].resize(self.val_batch_size,
                                                              *self.val_target_loader.shape[1:]).detach().tolist()

        with torch.no_grad():
            outputs = self.model(*data_features).resize(self.val_batch_size,
                                                        *self.val_target_loader.shape[1:]).detach().tolist()

        inputs = [data_features[i].resize(self.val_batch_size, *self.val_features_loaders[i].shape[1:],
                                        ).detach().tolist() for i in range(len(self.val_features_loaders))]

        images = []

        axes[0, 0].set_title("Real", pad=20)
        axes[0, 1].set_title("Pred", pad=20)
        for i in range(len(self.val_features_loaders)):
            axes[0, 2 + i].set_title(f"Input {i}", pad=20)

        for k in range(self.val_batch_size):

            images.append(axes[k, 0].imshow(data_target[k], cmap="jet"))
            images.append(axes[k, 1].imshow(outputs[k], cmap="jet"))

            for i in range(len(self.val_features_loaders)):
                images.append(axes[k, 2 + i].imshow(inputs[i][k], cmap="jet"))

        for im in images:
            fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, format=format)

        plt.tight_layout()
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

        for row in axes:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.show()

        plt.clf()
        matplotlib.pyplot.close()


    def train(self, epochs, criterion=None, optimizer=None, scheduler=None, step_saving=True, step_plotting=True, callbacks=[]):

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

            print('Epoch: {} \tTraining Loss: {:.9f} \tValidating Loss: {:.9f} \tTime: {:.2f} m'.
                  format(epoch + 1, train_loss,test_loss, (time.time() - start_point) / 60,2))

            if step_saving:
                if len(self.test_losses) and test_loss < min(self.test_losses):
                    self.save(onnx=False, pth=False)

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            if scheduler:
                scheduler.step()

            for callback in callbacks:
                callback()


            print("\n", "=" * 100, "\n", sep="")
            gc.collect()




def get_loss():
    return torch.nn.MSELoss()

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)



if __name__ == '__main__':
    pass
