import os
import numpy as np
import torch

from common.prelude import (read_matrix, rescale_array, arrayToCsv, array_save)
from common.stream import printProgressBar


def basic_test(data_path, width, height, save_path, model, device, csv=True, txt=True, png=True):
    noised_dir = os.path.join(data_path, "noised", "txt", )
    clear_dir = os.path.join(data_path, "clear", "txt")
    noised_files = [os.path.join(noised_dir, f) for f in os.listdir(noised_dir) if
                    os.path.isfile(os.path.join(noised_dir, f))]
    clear_files = [os.path.join(clear_dir, f) for f in os.listdir(clear_dir) if
                   os.path.isfile(os.path.join(clear_dir, f))]
    test_dir = os.path.join(save_path, "png")

    assert len(clear_files) == len(noised_files)

    print(f"Testing data: {data_path}")

    if png:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    with open(os.path.join(save_path, "Report.txt"), "w", encoding="utf8") as report_file:

        for (i, (noised_fp, clear_fp)) in enumerate(zip(noised_files, clear_files)):

            noised_name, clear_name = os.path.basename(noised_fp), os.path.basename(clear_fp)
            new_name = ".".join(noised_name.split(".")[:-1])

            assert noised_name == clear_name

            noised_arr = read_matrix(noised_fp)
            noised_arr_scaled = rescale_array(noised_arr)

            res_neuro_arr = rescale_array(model(
                torch.tensor(noised_arr_scaled, dtype=torch.float, device=device).resize(1, 1, width, height)
            ).resize(width, height).detach().cpu().numpy(), to=(np.min(noised_arr), np.max(noised_arr)))

            if csv:
                csv_dir = os.path.join(save_path, "csv", "test", f"{new_name}.csv")
                # if not os.path.exists(os.path.dirname(csv_dir)):
                #     os.makedirs(os.path.dirname(csv_dir))
                arrayToCsv(res_neuro_arr, csv_dir)
            if txt:
                txt_dir = os.path.join(save_path, "txt", "test", f"{new_name}.xls")
                # if not os.path.exists(os.path.dirname(txt_dir)):
                #     os.makedirs(os.path.dirname(txt_dir))
                array_save(res_neuro_arr, txt_dir)

            clear_arr = read_matrix(clear_fp)

            print(f"{i}) Diff:", np.mean(np.abs(res_neuro_arr - clear_arr)), file=report_file)

            if png:
                axes[0].imshow(noised_arr)
                axes[1].imshow(res_neuro_arr)
                axes[2].imshow(clear_arr)
                png_data_path = os.path.join(test_dir, f"{new_name}.png")
                if not os.path.exists(test_dir):
                    os.makedirs(test_dir)
                fig.savefig(png_data_path)

            printProgressBar(i + 1, len(noised_files), prefix='Progress:', suffix='Complete', length=50)
        print()