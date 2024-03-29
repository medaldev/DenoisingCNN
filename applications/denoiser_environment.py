import datetime
import os
import random

import torch

import common.fstream
import datageneration.generators
import models
from applications import model_manager
from common.fstream import (create_dir_of_file_if_not_exists, create_dir_if_not_exists)
from dataloaders import SimpleLoader2d
from testing.basic_test import basic_test
from datetime import datetime


class DenoiserEnvironment:

    def __init__(self, name_model, name_dataset, path_base, device="cuda"):
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.path_base = path_base
        self.model = None
        self.device = torch.device(device)

        self.path_save_train_plots = os.path.join(path_base, "runs", "train_plots", name_dataset,
                                                  name_model)

        self.path_train = os.path.join(path_base, "data", "datasets", name_dataset, "train")
        self.path_test = os.path.join(path_base, "data", "datasets", name_dataset, "val")

        self.path_train_noisy = os.path.join(self.path_train, "noised")
        self.path_test_noisy = os.path.join(self.path_test, "noised")

        self.path_train_normal = os.path.join(self.path_train, "clear")
        self.path_test_normal = os.path.join(self.path_test, "clear")

        self.train_losses, self.test_losses = [], []

    def path_save_model(self, model_type):
        return os.path.join(self.path_base, "assets", model_type,
                            f"{self.name_model}.{model_type}")

    def load_model(self, model_type="pt"):
        try:
            self.model = torch.load(self.path_save_model(model_type)).to(self.device)

        except Exception as e:
            self.init_model()
            print("Error when loading pretrained model. Use custom.", e)

    def init_model(self, model_class):
        self.model = model_class().to(self.device)
        print("New model created.")

    def load_data(self, width, height, batch_size, read_tensor_x=None, read_tensor_y=None):
        if read_tensor_x is None:
            read_tensor_x = common.fstream.read_tensor
        if read_tensor_y is None:
            read_tensor_y = common.fstream.read_tensor

        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.train_noisy_loader = SimpleLoader2d(self.path_train_noisy, self.device, batch_size, width, height,
                                                 read_tensor=read_tensor_x)
        self.train_normal_loader = SimpleLoader2d(self.path_train_normal, self.device, batch_size, width, height,
                                                  read_tensor=read_tensor_y)

        self.val_noisy_loader = SimpleLoader2d(self.path_test_noisy, self.device, batch_size, width, height,
                                               read_tensor=read_tensor_x)
        self.val_normal_loader = SimpleLoader2d(self.path_test_normal, self.device, batch_size, width, height,
                                                read_tensor=read_tensor_y)

        self.score_noisy_loader = SimpleLoader2d(self.path_test_noisy, self.device, 1, width, height,
                                                 read_tensor=read_tensor_x)
        self.score_normal_loader = SimpleLoader2d(self.path_test_normal, self.device, 1, width, height,
                                                  read_tensor=read_tensor_y)

    def load_equal_data(self, width, height, batch_size, read_tensor_x=None, read_tensor_y=None):
        if read_tensor_x is None:
            read_tensor_x = common.fstream.read_tensor
        if read_tensor_y is None:
            read_tensor_y = common.fstream.read_tensor

        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.train_noisy_loader = SimpleLoader2d(self.path_train_normal, self.device, batch_size, width, height,
                                                 read_tensor=read_tensor_x)
        self.train_normal_loader = SimpleLoader2d(self.path_train_normal, self.device, batch_size, width, height,
                                                  read_tensor=read_tensor_y)

        self.val_noisy_loader = SimpleLoader2d(self.path_test_normal, self.device, batch_size, width, height,
                                               read_tensor=read_tensor_x)
        self.val_normal_loader = SimpleLoader2d(self.path_test_normal, self.device, batch_size, width, height,
                                                read_tensor=read_tensor_y)

        self.score_noisy_loader = SimpleLoader2d(self.path_test_normal, self.device, 1, width, height,
                                                 read_tensor=read_tensor_x)
        self.score_normal_loader = SimpleLoader2d(self.path_test_normal, self.device, 1, width, height,
                                                  read_tensor=read_tensor_y)

    def generate_data(self, name, data_goal, n, width, height, cell_size, pct_noise=None, k0=None, csv=True, txt=True, png=True, parameters=None,
                      verbose=True, generator_class=None):
        if generator_class:
            generator = generator_class()
        else:
            generator = datageneration.generators.get_basic_generator()

        generator.generate_dataset(f"{name}/{data_goal}", n, width, height, cell_size, pct_noise, k0, csv, txt, png, parameters,
                                   verbose)

    def clear_metrics(self):
        self.train_losses, self.test_losses = [], []

    def plot_step_results(self, limit=1, op_count=1, batch_size=None, cmap="viridis"):
        if not batch_size:
            batch_size = self.batch_size
        self.plot_batches(self.path_save_train_plots, self.train_noisy_loader, self.train_normal_loader, "train",
                          limit=limit, op_count=op_count, cmap=cmap)
        self.plot_batches(self.path_save_train_plots, self.val_noisy_loader, self.val_normal_loader, "test",
                          limit=limit, op_count=op_count, cmap=cmap)
        print(f"Step results plotted to {self.path_save_train_plots}.")

    def train(self, epochs, criterion=None, optimizer=None, step_saving=True, step_plotting=True):
        train_hist = model_manager.train(self.model, self.train_noisy_loader, self.train_normal_loader,
                                           self.val_noisy_loader,
                                           self.val_normal_loader,
                                           epochs, self.device,
                                           path_save=self.path_save_model("pt") if step_saving else None,
                                           optimizer=optimizer,
                                           criterion=criterion,
                                           callbacks=[self.plot_step_results] if step_plotting else [])
        self.train_losses += train_hist[0]
        self.test_losses += train_hist[1]

    def show_metrics(self, n_last=None):
        import matplotlib.pyplot as plt
        plt.close()
        plt.cla()
        plt.clf()
        plt.plot(self.train_losses if n_last is None else self.train_losses[-n_last:], label="train_loss")
        plt.plot(self.test_losses if n_last is None else self.test_losses[-n_last:], label="test_loss", color="orange")
        plt.legend()
        plt.title("Loss metric")
        plt.show()

    def save(self, onnx=False, pth=False):
        model_manager.save_full_model(self.model, self.path_save_model("pt"))
        model_manager.save_traced_model(self.model, self.path_save_model("pt_traced"))

        if pth:
            path_save_model_pth = self.path_save_model("pth")
            inp = torch.randn((1, 1, self.width, self.height), device=self.device)
            with torch.no_grad():
                traced_cell = torch.jit.trace(self.model, inp)

            create_dir_of_file_if_not_exists(path_save_model_pth)
            torch.jit.save(traced_cell, path_save_model_pth)

        if onnx:
            path_save_model_onnx = self.path_save_model("onnx")
            inp = torch.randn((1, 1, self.width, self.height), device=self.device)
            create_dir_of_file_if_not_exists(path_save_model_onnx)
            model_manager.save_onnx_model(self.model, path_save_model_onnx, inp)

    def show_single(self, concrete=None, op_count=1, figsize=(18, 8), cmap="viridis"):
        if concrete is None:
            concrete = random.randint(0, len(self.val_noisy_loader) - 1)
        print(f"Show example №{concrete}")
        data_noisy, data_normal = self.val_noisy_loader[concrete], self.val_normal_loader[concrete]
        self.plot_batch(data_noisy, data_normal, op_count=op_count, figsize=figsize, cmap=cmap)

    def plot_batch(self, data_noisy, data_normal, op_count=1, png_save_path=None, figsize=(18, 8), cmap="viridis"):
        images_noisy, _ = data_noisy
        images_noisy = images_noisy.to(self.device)

        images_normal, __ = data_normal
        images_normal = images_normal.to(self.device)

        outputs = self.model(images_noisy)

        import matplotlib.pyplot as plt
        if png_save_path is not None:
            plt.ioff()
            plt.tight_layout()
        fig, axes = plt.subplots(self.batch_size, 2 + op_count, figsize=figsize)

        for k in range(self.batch_size):
            image_noisy = images_noisy[k].resize(self.width, self.height).tolist()
            image_normal = images_normal[k].resize(self.width, self.height).tolist()

            im1 = axes[k, 0].imshow(image_noisy, cmap=cmap)
            im2 = axes[k, 1 + op_count].imshow(image_normal, cmap=cmap)
            axes[k, 0].set_axis_off()
            axes[k, 1 + op_count].set_axis_off()

            cb1 = fig.colorbar(im1, orientation='vertical', fraction=0.046, pad=0.02)
            cb2 = fig.colorbar(im2, orientation='vertical', fraction=0.046, pad=0.02)
            cb1.ax.yaxis.set_tick_params(color="white")
            cb2.ax.yaxis.set_tick_params(color="white")

        for j in range(op_count):
            for k in range(self.batch_size):
                image_out = outputs[k].resize(self.width, self.height).detach().tolist()
                im3 = axes[k, 1 + j].imshow(image_out, cmap=cmap)
                axes[k, 1 + j].set_axis_off()
                cb3 = fig.colorbar(im3, orientation='vertical', fraction=0.046, pad=0.02)
                cb3.ax.yaxis.set_tick_params(color="white")

            if j + 1 < op_count:
                outputs = self.model(outputs)
        #plt.subplots_adjust(wspace=0, hspace=10)

        if png_save_path is not None:
            if not os.path.exists(os.path.dirname(png_save_path)):
                os.makedirs(os.path.dirname(png_save_path))
            fig.savefig(png_save_path)
            plt.clf()
            plt.cla()

            fig.clf()
            plt.close()
        else:
            for j in range(op_count):
                for k in range(self.batch_size):
                        axes[k, j + 0].tick_params(colors="white")
                        axes[k, j + 1].tick_params(colors="white")
                        axes[k, j + 2].tick_params(colors="white")

            fig.patch.set_alpha(0.0)
            plt.show()

    def plot_batches(self, directory, noisy_loader, normal_loader, prefix, limit, op_count=1, cmap="viridis"):

        assert op_count > 0  # кол-во раз обработки изображения моделью
        with torch.no_grad():
            i = 0
            for data_noisy, data_normal in zip(noisy_loader, normal_loader):
                png_save_path = os.path.join(directory, prefix,
                                             f"BATCH_ELEMENT_{prefix}_res{i}.png")
                self.plot_batch(data_noisy, data_normal, op_count=op_count, png_save_path=png_save_path, cmap=cmap)
                i += 1
                if i >= limit:
                    break

    def score(self, show=False):

        scores_before = []
        scores = []

        with torch.no_grad():
            for data_noisy, data_normal in zip(self.score_noisy_loader, self.score_normal_loader):
                images_noisy, _ = data_noisy
                images_noisy = images_noisy.to(self.device)

                images_normal, __ = data_normal
                images_normal = images_normal.to(self.device)

                outputs = self.model(images_noisy)

                scores.append(torch.mean(torch.abs(outputs - images_normal)).tolist())
                scores_before.append(torch.mean(torch.abs(images_noisy - images_normal)).tolist())

            print()

        if show:
            import matplotlib.pyplot as plt

            plt.hist(scores_before, alpha=0.5, label='noise before')
            plt.hist(scores, alpha=0.5, label='noise after')
            plt.legend(loc='upper right')
            plt.grid()
            plt.show()

        return scores_before, scores

    def test_on_dataset(self, include_train=True, include_test=True, csv=True, txt=True, png=True, cmap="viridis"):
        base_test_path = os.path.join(self.path_base, "runs", "tests", self.name_model,
                                      f"{self.name_dataset}", str(datetime.now()))

        report_train_path = os.path.join(base_test_path, "train")
        report_test_val_path = os.path.join(base_test_path, "val")

        create_dir_if_not_exists(report_train_path)
        create_dir_if_not_exists(report_test_val_path)

        if include_train:
            basic_test(self.path_train, self.width, self.height, report_train_path, self.model, self.device, csv, txt, png, cmap=cmap)

        if include_test:
            basic_test(self.path_test, self.width, self.height, report_test_val_path, self.model, self.device, csv, txt,
                   png, cmap=cmap)

        print("Testing ended.")


if __name__ == '__main__':
    env = DenoiserEnvironment(name_model="gcg2_model_6", name_dataset="gcg4",
                              path_base="/home/amedvedev/fprojects/python/denoising")
    env.load_model(model_type="pth")
    env.load_data(100, 100, 2, read_tensor=common.fstream.read_tensor)

    #
    # env.test_on_dataset()
    # env.train(3)
    # env.show_metrics()
    # env.generate_data("gcg3", "val", n=5, width=80, height=80, cell_size=2, csv=True, txt=True, png=True)
    env.score(show=True)
    env.test_on_dataset()
    #
    # # env.init_model()
    # env.save(pth=True)
