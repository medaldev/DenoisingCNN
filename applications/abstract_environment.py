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


class AbstractEnvironment:

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

    def clear_metrics(self):
        self.train_losses, self.test_losses = [], []

    def show_metrics(self, n_last=None, train=True, val=False):
        import matplotlib.pyplot as plt
        plt.close()
        plt.cla()
        plt.clf()
        if train:
            plt.plot(self.train_losses if n_last is None else self.train_losses[-n_last:], label="train_loss")
        if val:
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

    def save_pth(self, inp_example):
        path_save_model_pth = self.path_save_model("pth")
        with torch.no_grad():
            traced_cell = torch.jit.trace(self.model, inp_example)

        create_dir_of_file_if_not_exists(path_save_model_pth)
        torch.jit.save(traced_cell, path_save_model_pth)

    def save_onnx(self, inp_example):
        path_save_model_onnx = self.path_save_model("onnx")
        create_dir_of_file_if_not_exists(path_save_model_onnx)
        model_manager.save_onnx_model(self.model, path_save_model_onnx, inp_example)