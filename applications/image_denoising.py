import torch
from torch.utils.data.dataloader import default_collate

from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import os
import time

from models import ConvAutoencoder


def main():
    device = 'cpu' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 16
    EPOCHS = 5

    path_base = "/home/amedvedev/fprojects/data/"
    path_save_model = "../assets/pt/img_denoise.pt"
    path_dataset = "rain800_idcgan/"

    path_train = os.path.join(path_base, path_dataset, "training")
    path_test = os.path.join(path_base, path_dataset, "test")

    path_train_rainy = path_train + "_rainy"
    path_test_rainy = path_test + "_rainy"

    path_train_normal = path_train + "_normal"
    path_test_normal = path_test + "_normal"

    model = ConvAutoencoder().to(device)

    (train_noisy_loader, train_normal_loader), (val_noisy_loader, val_normal_loader) = get_loaders(
        path_train_rainy, path_test_rainy, path_train_normal, path_test_normal, BATCH_SIZE, device
    )

    train(model, train_noisy_loader, train_normal_loader, EPOCHS, device, path_save=path_save_model)


def get_transformations():
    transformations = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(235),
        transforms.CenterCrop(230),
        # transforms.FiveCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transformations


def get_loaders(path_train_noisy, path_test_noisy, path_train_normal, path_test_normal, batch_size, device,
                transformations=None):
    if not transformations:
        transformations = get_transformations()

    train_noisy_set = datasets.ImageFolder(path_train_noisy, transform=transformations)
    val_noisy_set = datasets.ImageFolder(path_test_noisy, transform=transformations)

    train_normal_set = datasets.ImageFolder(path_train_normal, transform=transformations)
    val_normal_set = datasets.ImageFolder(path_test_normal, transform=transformations)

    # Loaders

    train_noisy_loader = torch.utils.data.DataLoader(train_noisy_set, batch_size=batch_size, shuffle=False)
    val_noisy_loader = torch.utils.data.DataLoader(val_noisy_set, batch_size=batch_size, shuffle=False)

    train_normal_loader = torch.utils.data.DataLoader(train_normal_set, batch_size=batch_size, shuffle=False)

    val_normal_loader = torch.utils.data.DataLoader(val_normal_set, batch_size=batch_size, shuffle=False)

    return (train_noisy_loader, train_normal_loader), (val_noisy_loader, val_normal_loader)


def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)


def get_loss():
    return torch.nn.MSELoss()


def train(model, train_noisy_loader, train_normal_loader, test_noisy_loader, test_normal_loader,
          n_epochs, device, path_save=None, criterion=None, optimizer=None, callbacks=[]):

    for cb in callbacks:
        cb()

    # Loss function
    if not criterion:
        criterion = get_loss()

    # Optimizer
    if not optimizer:
        optimizer = get_optimizer(model)

    loss_values = []
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        test_loss = 0.0

        itr, itr_test = 0, 0
        start_point = time.time()
        # Training

        for data_noisy, data_normal in zip(train_noisy_loader, train_normal_loader):
            images_noisy, _ = data_noisy
            images_noisy = images_noisy.to(device)

            images_normal, __ = data_normal
            images_normal = images_normal.to(device)

            optimizer.zero_grad()
            outputs = model(images_noisy)
            loss = criterion(outputs, images_normal)
            loss.backward()
            optimizer.step()

            loss_deltha = loss.item() * images_noisy.size(0)
            train_loss += loss_deltha / len(train_noisy_loader)
            itr += 1
            # print("-iteration", itr, "deltha_loss", loss_deltha/len(train_noisy_loader))
            printProgressBar(itr, len(train_noisy_loader), prefix = 'Training progress:', suffix = 'Complete', length = 50)
        print()

        with torch.no_grad():
            for data_noisy, data_normal in zip(test_noisy_loader, test_normal_loader):
                images_noisy, _ = data_noisy
                images_noisy = images_noisy.to(device)

                images_normal, __ = data_normal
                images_normal = images_normal.to(device)

                outputs = model(images_noisy)
                loss = criterion(outputs, images_normal)

                loss_deltha = loss.item() * images_noisy.size(0)
                test_loss += loss_deltha / len(test_noisy_loader)
                itr_test += 1
                printProgressBar(itr_test, len(test_noisy_loader), prefix = 'Validating progress:', suffix = 'Complete', length = 50)
            print()

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidating Loss: {:.6f} \tTime: {:.2f} m'.format(epoch + 1, train_loss, test_loss, (time.time() - start_point) / 60, 2))

        if path_save:
            if len(loss_values) and train_loss < min(loss_values):
                save_full_model(model, path_save)
                print(f"Model saved successfully at {path_save}.")

        loss_values.append(train_loss)

        for cb in callbacks:
            cb()

        print("\n", "=" * 100, "\n", sep="")


def save_full_model(model, path_model):
    torch.save(model, path_model)


def save_onnx_model(model, path_model, device, image_size=(100, 100)):
    with torch.no_grad():
        inp = torch.randn((1, 3, image_size[0], image_size[1]), device=device)

        torch.onnx.export(model,
                          (inp,),
                          path_model,
                          verbose=True,
                          input_names=("img",),
                          output_names=("output",),
                          opset_version=14,
                          do_constant_folding=False,
                          export_params=True,
                          dynamic_axes=None)


def save_results(model, device, directory, rainy_loader, normal_loader, prefix, limit, nrow=5, op_count=1):

    assert op_count > 0 # кол-во раз обработки изображения моделью

    with torch.no_grad():
      i = 0
      for data_noisy, data_normal in zip(rainy_loader, normal_loader):
            images_noisy, _ = data_noisy
            images_noisy = images_noisy.to(device)

            images_normal, __ = data_normal
            images_normal = images_normal.to(device)

            outputs = model(images_noisy)
            for j in range(op_count):
                res_cat = torch.cat((images_noisy, outputs, images_normal))

                img = make_grid(res_cat, nrow = nrow) # метод делает сетку из картинок
                #img = img.numpy().transpose((1, 2, 0)) # транспонируем для отображения в картинке

                if not os.path.exists(directory):
                  os.makedirs(directory)
                save_image(img, os.path.join(directory, f"{prefix}_res{i}_op_count({j + 1}).png"))
                if op_count > 1:
                  outputs = model(outputs)


            i+= 1
            if i >= limit:
              break


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█',
                      printEnd = "\r", properties={}):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    prop_string = " | ".join([f"{k}: {v}" for k, v in properties.items()])
    print(f'\r{prefix} |{bar}| {percent}% {suffix} | {prop_string}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    main()