import os
import torch

import datageneration.generators
import models
from applications import image_denoising
from dataloaders import SimpleLoader2d


class DenoiserEnvironment:

    def __init__(self, name_model, name_dataset, path_base, device="cuda"):
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.path_base = path_base
        self.model = None
        self.path_save_model = os.path.join(path_base, "DenoisingCNN", "assets", "pt",
                                            f"{name_dataset}_{name_model}.pt")
        self.device = torch.device(device)

        self.path_save_train_plots = os.path.join(path_base, "DenoisingCNN", "runs", "train_plots", name_dataset,
                                                  name_model)

        self.path_train = os.path.join(path_base, "data", "datasets", name_dataset, "train")
        self.path_test = os.path.join(path_base, "data", "datasets", name_dataset, "val")

        self.path_train_noisy = self.path_train + "/noised"
        self.path_test_noisy = self.path_test + "/noised"

        self.path_train_normal = self.path_train + "/clear"
        self.path_test_normal = self.path_test + "/clear"

        self.train_losses, self.test_losses = [], []

    def load_model(self):
        try:
            self.model = torch.load(self.path_save_model).to(self.device)
        except Exception as e:
            self.init_model()
            print("Error when loading pretrained model. Use custom.")

    def init_model(self, model_class=None):
        if model_class:
            self.model = model_class().to(self.device)
        else:
            self.model = models.get_basic_model().to(self.device)
        print("New model created.")

    def load_data(self, width, height, batch_size, read_tensor=None):
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
                                           epochs, self.device, path_save=self.path_save_model, optimizer=optimizer,
                                           criterion=criterion,
                                           callbacks=[self.plot_step_results])
        self.train_losses += train_hist[0]
        self.test_losses += train_hist[1]

    def show_metrics(self):
        import matplotlib.pyplot as plt
        plt.plot(self.train_losses, label="train_loss")
        plt.plot(self.test_losses, label="test_loss", color="orange")
        plt.show()

    def save(self, onnx=False):
        image_denoising.save_full_model(self.model, self.path_save_model)

        if onnx:
            path_save_model_onnx = os.path.join(self.path_base, "DenoisingCNN", "assets", "onnx",
                                                f"{self.name_dataset}_{self.name_model}.onnx")
            inp = torch.randn((1, 1, self.width, self.height), device=self.device)
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
                        image_out = outputs[k].resize(self.width, self.height).detach().numpy()

                        axes[k, j + 0].imshow(image_noisy)
                        axes[k, j + 1].imshow(image_out)
                        axes[k, j + 2].imshow(image_normal)

                        png_save_path = os.path.join(directory, prefix,
                                                     f"BATCH_ELEMENT_{k}_{prefix}_res{i}_op_count({j + 1}).png")

                        if not os.path.exists(directory(png_save_path)):
                            os.makedirs(directory(png_save_path))

                        fig.savefig(png_save_path)

                    if op_count > 1:
                        outputs = self.model(outputs)

                i += 1
                if i >= limit:
                    break

    def score(self, model, show=False):

        scores_before = []
        scores = []

        with torch.no_grad():
            for data_noisy, data_normal in zip(self.score_noisy_loader, self.score_normal_loader):
                images_noisy, _ = data_noisy
                images_noisy = images_noisy.to(self.device)

                images_normal, __ = data_normal
                images_normal = images_normal.to(self.device)

                outputs = model(images_noisy)

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
