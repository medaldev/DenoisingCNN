import torch
from torch.utils.data.dataloader import default_collate

from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import os
import time

from common.stream import printProgressBar

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
    test_loss_values = []
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
            if len(test_loss_values) and test_loss < min(test_loss_values):
                save_full_model(model, path_save)
                print(f"Model saved successfully at {path_save}.")

        loss_values.append(train_loss)
        test_loss_values.append(test_loss)

        for cb in callbacks:
            cb()

        print("\n", "=" * 100, "\n", sep="")
    return loss_values, test_loss_values


def save_full_model(model, path_model):

    torch.save(model, path_model)

def save_traced_model(model, path_model):
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path_model)



def save_onnx_model(model, path_model, inp):
    with torch.no_grad():

        torch.onnx.export(model,
                          (inp,),
                          path_model,
                          verbose=True,
                          # input_names=("img",),
                          # output_names=("output",),
                          opset_version=14,
                          do_constant_folding=False,
                          export_params=True,
                          dynamic_axes=None)
