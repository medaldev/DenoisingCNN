import argparse
import os
import sys

from torch import nn

from pathlib import Path

PATH_BASE = str(Path(os.path.realpath(__file__)).parent.parent.absolute())

print("PATH_BASE", PATH_BASE)

sys.path.append(PATH_BASE)

from scripts.real_model import DenoiserModel
from applications.poly_features_environment import PolyFeaturesEnv
import torch
import models
import numpy as np
from common.fstream import read_tensor, read_matrix, rescale_array, read_mc_tensor
from common.matrix import add_noise
from loguru import logger


def load(path):
    re_im = np.fromfile(path, dtype=np.double)
    x = np.vectorize(complex)(re_im[::2], re_im[1::2])
    return x.real


def get_noised(path, pct):
    uvych_noised = load(path)
    for i in range(len(uvych_noised)):
        ppp = np.random.uniform(low=0, high=pct)
        uvych_noised[i] *= (1 + ppp)

    return uvych_noised


def get_few_noised(path, pct, k):
    return np.vstack([get_noised(path, pct) for _ in range(k)])


def get_mean_noised(path, pct, k):
    return sum([get_noised(path, pct) for _ in range(k)]) / k


def my_loss(output, target, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    f = lambda x: torch.fft.fftn(x, dim=(-3, -2, -1))

    # Frequency domain loss
    abs_diff_freq = torch.abs(f(output) - f(target))

    loss = torch.mean(abs_diff_freq) + torch.mean(torch.abs((output) - (target))) + torch.max(
        torch.abs((output) - (target)))
    return loss


def test(env):
    losses = []
    errors = []
    init_errors = []
    losses_dataset = []
    with torch.no_grad():

        for row in zip(*env.val_features_loaders + [env.val_target_loader]):
            data_features = list(row)
            data_target = data_features.pop()
            outputs = env.model(*data_features)  # * 65.

            for ex_id in range(env.val_batch_size):
                # losses_dataset_step = torch.max(torch.abs(data_features[0][ex_id] - data_target[ex_id])).detach().tolist()
                # losses_dataset.append(losses_dataset_step)

                loss = torch.max(torch.abs(outputs[ex_id] - data_target[ex_id]))
                losses.append(loss.detach().tolist())
                # logger.info(data_target[ex_id].size())

                # init_error = torch.mean(torch.abs(data_features[0][ex_id] - data_target[ex_id]) / torch.abs(data_target[ex_id]))
                # init_errors.append(init_error.detach().tolist())

                error = torch.mean(torch.abs(outputs[ex_id] - data_target[ex_id]) / torch.abs(data_target[ex_id]))
                errors.append(error.detach().tolist())

    # logger.info("Начальное среднее отклонение по значениям:", sum(losses_dataset) / len(losses_dataset))
    logger.info(f"Текущее среднее отклонение по значениям: {sum(losses) / len(losses)}")

    # logger.info("Начальное максимальное отклонение по значениям:", max(losses_dataset))
    logger.info(f"Текущее максимальное отклонение по значениям в векторе: {max(losses)}")

    # logger.info()
    # logger.info("Начальная средняя относительная ошибка:", sum(init_errors) / len(init_errors))
    logger.info(f"Текущая средняя относительная ошибка: {sum(errors) / len(errors)}")
    # logger.info()
    # logger.info("Начальная максимальная относительная ошибка:", max(init_errors))
    # logger.info("Текущая максимальная относительная ошибка:", max(errors))


def main(
        name_dataset, count_evych, pct_noise, epochs,
        model_name, load_from_pretrain,
        train_batch_size, val_batch_size,
        learning_rate,
        num_par_filters=None,
        num_denoiser_blocks=None,
        device="cpu"
):
    NAME_DATASET = name_dataset
    env = PolyFeaturesEnv(name_model=model_name,
                          name_dataset=NAME_DATASET,
                          path_base=PATH_BASE,
                          device_name=device)

    dtype = torch.double

    env.clear_features_and_targets()
    env \
        .set_batch_size(
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size
    ) \
        .load_feature(
        shape=(count_evych * 3, 10, 10, 10), feature_name="Evych",
        mapper=lambda path: get_few_noised(path, pct_noise, count_evych), transform=None, dtype=dtype
    ) \
        .set_target(
        shape=(3, 10, 10, 10), target_name="Evych", mapper=load, transform=None, dtype=dtype
    )

    logger.debug(f"Train/Val selections counts: {env.train_count}, {env.val_count}")

    print(load_from_pretrain)
    if load_from_pretrain:
        env.model = torch.load(env.path_save_model("pt"), map_location=env.device, weights_only=True).to(env.device)
        print("Model loaded from pretrained")
    else:
        env.model = DenoiserModel(in_channels=3 * count_evych, num_par_filters=num_par_filters,
                                  num_denoiser_blocks=num_denoiser_blocks).to(env.device, dtype=dtype)

        def weights_init(m):
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data)

        env.model.apply(weights_init)

    opt = torch.optim.Adam(env.model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4)

    env.train(epochs, step_saving=True, step_plotting=False,
              optimizer=opt, scheduler=None,
              criterion=my_loss,
              callbacks=[
                  lambda: scheduler.step(env.test_losses[-1]),
                  lambda: test(env)
              ])

    env.show_metrics(n_last=epochs, train=False, val=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name_dataset", type=str, required=True)
    parser.add_argument("--count_evych", type=int, required=True, help="Count of evych.")
    parser.add_argument("--pct_noise", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train the model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--load_from_pretrain", type=bool, required=False, default=False,
                        help="Whether to load from a pretrained model (True/False).")
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--val_batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_par_filters", type=int, required=True)
    parser.add_argument("--num_denoiser_blocks", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(
        name_dataset=args.name_dataset,
        count_evych=args.count_evych,
        pct_noise=args.pct_noise,
        epochs=args.epochs,
        model_name=args.model_name,
        load_from_pretrain=args.load_from_pretrain,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        num_par_filters=args.num_par_filters,
        num_denoiser_blocks=args.num_denoiser_blocks,
        device=args.device,
    )
