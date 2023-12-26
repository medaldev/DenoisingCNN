import os
import torch

import common.fstream
import datageneration.generators
import models
from applications import image_denoising
from common.fstream import create_dir_of_file_if_not_exists
from dataloaders import SimpleLoader2d


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

        self.path_train_noisy = self.path_train + "/noised"
        self.path_test_noisy = self.path_test + "/noised"

        self.path_train_normal = self.path_train + "/clear"
        self.path_test_normal = self.path_test + "/clear"

        self.train_losses, self.test_losses = [], []
        
    def path_save_model(self, model_type):
        return os.path.join(self.path_base, "assets", model_type,
                     f"{self.name_dataset}_{self.name_model}.{model_type}")

    def load_model(self, model_type="pt"):
        try:
            self.model = torch.load(self.path_save_model(model_type)).to(self.device)
                
        except Exception as e:
            self.init_model()
            print("Error when loading pretrained model. Use custom.", e)

    def init_model(self, model_class=None):
        if model_class:
            self.model = model_class().to(self.device)
        else:
            self.model = models.get_basic_model().to(self.device)
        print("New model created.")

    def load_data(self, width, height, batch_size, read_tensor=None):
        if read_tensor is None:
            read_tensor = common.fstream.read_tensor
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.train_noisy_loader = SimpleLoader2d(self.path_train_noisy, self.device, batch_size, width, height,
                                                 read_tensor=read_tensor)
        self.train_normal_loader = SimpleLoader2d(self.path_train_normal, self.device, batch_size, width, height,
                                                  read_tensor=read_tensor)

        self.val_noisy_loader = SimpleLoader2d(self.path_test_noisy, self.device, batch_size, width, height,
                                               read_tensor=read_tensor)
        self.val_normal_loader = SimpleLoader2d(self.path_test_normal, self.device, batch_size, width, height,
                                                read_tensor=read_tensor)

        self.score_noisy_loader = SimpleLoader2d(self.path_test_noisy, self.device, 1, width, height,
                                                 read_tensor=read_tensor)
        self.score_normal_loader = SimpleLoader2d(self.path_test_normal, self.device, 1, width, height,
                                                  read_tensor=read_tensor)

    def generate_data(self, name, n, width, height, cell_size, csv=True, txt=True, png=True, parameters=None,
                         verbose=True, generator_class=None):
        if generator_class:
            generator = generator_class()
        else:
            generator = datageneration.generators.get_basic_generator()

        generator.generate_dataset(name, n, width, height, cell_size, csv, txt, png, parameters, verbose)

    def clear_metrics(self):
        self.train_losses, self.test_losses = [], []

    def plot_step_results(self, limit=1, op_count=1, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        self.test_plot(self.path_save_train_plots, self.train_noisy_loader, self.train_normal_loader, "train",
                       limit=limit, nrow=batch_size, op_count=op_count)
        self.test_plot(self.path_save_train_plots, self.val_noisy_loader, self.val_normal_loader, "test",
                       limit=limit, nrow=batch_size, op_count=op_count)
        print(f"Step results plotted to {self.path_save_train_plots}.")

    def train(self, epochs, criterion=None, optimizer=None):
        train_hist = image_denoising.train(self.model, self.train_noisy_loader, self.train_normal_loader,
                                           self.val_noisy_loader,
                                           self.val_normal_loader,
                                           epochs, self.device, path_save=self.path_save_model("pt"), optimizer=optimizer,
                                           criterion=criterion,
                                           callbacks=[self.plot_step_results])
        self.train_losses += train_hist[0]
        self.test_losses += train_hist[1]

    def show_metrics(self):
        import matplotlib.pyplot as plt
        plt.plot(self.train_losses, label="train_loss")
        plt.plot(self.test_losses, label="test_loss", color="orange")
        plt.show()

    def save(self, onnx=False, pth=False):
        image_denoising.save_full_model(self.model, self.path_save_model("pt"))

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
            image_denoising.save_onnx_model(self.model, path_save_model_onnx, inp)

    def test_plot(self, directory, noisy_loader, normal_loader, prefix, limit, nrow=5, op_count=1):

        assert op_count > 0  # кол-во раз обработки изображения моделью
        with torch.no_grad():
            i = 0
            for data_noisy, data_normal in zip(noisy_loader, normal_loader):
                images_noisy, _ = data_noisy
                images_noisy = images_noisy.to(self.device)

                images_normal, __ = data_normal
                images_normal = images_normal.to(self.device)

                outputs = self.model(images_noisy)

                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(self.batch_size, 3 * op_count, figsize=(18, 8))

                for j in range(op_count):
                    for k in range(self.batch_size):
                        image_noisy = images_noisy[k].resize(self.width, self.height).tolist()
                        image_normal = images_normal[k].resize(self.width, self.height).tolist()
                        image_out = outputs[k].resize(self.width, self.height).detach().tolist()

                        axes[k, j + 0].imshow(image_noisy)
                        axes[k, j + 1].imshow(image_out)
                        axes[k, j + 2].imshow(image_normal)

                    if op_count > 1:
                        outputs = self.model(outputs)

                png_save_path = os.path.join(directory, prefix,
                                             f"BATCH_ELEMENT_{prefix}_res{i}.png")

                if not os.path.exists(os.path.dirname(png_save_path)):
                    os.makedirs(os.path.dirname(png_save_path))
                fig.savefig(png_save_path)

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

            plt.hist(scores_before, alpha=0.5, label='x')
            plt.hist(scores, alpha=0.5, label='y')
            plt.legend(loc='upper right')
            plt.grid()
            plt.show()

        return scores_before, scores


if __name__ == '__main__':
    env = DenoiserEnvironment(name_model="model_5", name_dataset="gcg2", path_base="/home/amedvedev/fprojects/python/denoising")
    env.load_model(model_type="pt")
    env.load_data(100, 100, 4, read_tensor=common.fstream.read_tensor)
    #
    env.score(show=True)
    # env.train(10)
    # env.show_metrics()
    # # env.generate_data("gcg3", n=5, width=80, height=80, cell_size=2, csv=True, txt=True, png=True)
    # env.score(show=True)
    #
    # # env.init_model()
    # env.save(pth=True)


