import numpy as np
import torch

from datetime import datetime
import os









def basic_test(save_path, width, height, cell_size, model_path, device='cpu', csv=True, txt=True, png=True):
    noised_dir = os.path.join(save_path, "txt", "noised")
    clear_dir = os.path.join(save_path, "txt", "clear")
    noised_files = [os.path.join(noised_dir, f) for f in os.listdir(noised_dir) if
                      os.path.isfile(os.path.join(noised_dir, f))]
    clear_files = [os.path.join(clear_dir, f) for f in os.listdir(clear_dir) if
                    os.path.isfile(os.path.join(clear_dir, f))]
    test_dir = os.path.join(save_path, "test4")

    print(len(clear_files), len(noised_files))

    device = torch.device(device)
    model = torch.load(model_path).to(device).eval()

    if png:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for (i, (noised_fp, clear_fp)) in enumerate(zip(noised_files, clear_files)):

        noised_name, clear_name = os.path.basename(noised_fp), os.path.basename(clear_fp)
        new_name = ".".join(noised_name.split(".")[:-1])

        assert noised_name == clear_name

        noised_arr = read_matrix(noised_fp)
        noised_arr_scaled = rescale_array(noised_arr)

        res_neuro_arr = rescale_array(model(
            torch.tensor(noised_arr_scaled, dtype=torch.float, device=device).resize(1, 1, width, height)
        ).resize(width, height).detach().numpy(), to=(np.min(noised_arr), np.max(noised_arr)))

        if csv:
            csv_dir = os.path.join(save_path, "csv", "test",  f"{new_name}.csv")
            # if not os.path.exists(os.path.dirname(csv_dir)):
            #     os.makedirs(os.path.dirname(csv_dir))
            arrayToCsv(res_neuro_arr, csv_dir)
        if txt:
            txt_dir = os.path.join(save_path, "txt", "test", f"{new_name}.xls")
            # if not os.path.exists(os.path.dirname(txt_dir)):
            #     os.makedirs(os.path.dirname(txt_dir))
            array_save(res_neuro_arr, txt_dir)

        clear_arr = read_matrix(clear_fp)

        print(f"{i}) Diff:", np.mean(np.abs(res_neuro_arr - clear_arr)))

        if png:
            axes[0].imshow(noised_arr)
            axes[1].imshow(res_neuro_arr)
            axes[2].imshow(clear_arr)
            png_save_path = os.path.join(test_dir, f"{new_name}.png")
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            fig.savefig(png_save_path)



def generate_example(save_path, width, height, cell_size, csv=True, txt=True, png=True):

    surface = gen_data_polygons_1(width, height, cell_size, 5)
    #surface = gen_data_polygons_3(width, height, cell_size, 2, poly_radius=random.randint(4, 8))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    arr = surface.matrix()
    if np.sum(arr) > 0:


        this_label = str(datetime.now())
        if csv:
            arrayToCsv(arr, os.path.join(save_path, "csv", "clear", f"{this_label}.csv"))
        if txt:
            array_save(arr, os.path.join(save_path, "txt", "clear", f"{this_label}.xls"))

        if png:
            axes[0].imshow(arr)

        add_noise(arr, 2)
        # print_matrix(arr)
        if csv:
            arrayToCsv(arr, os.path.join(save_path, "csv", "noised",  f"{this_label}.csv"))
        if txt:
            array_save(arr, os.path.join(save_path, "txt", "noised", f"{this_label}.xls"))


        if png:
            axes[1].imshow(arr)
            png_save_path = os.path.join(save_path, "png", f"{this_label}_noised.png")
            if not os.path.exists(os.path.dirname(png_save_path)):
                os.makedirs(os.path.dirname(png_save_path))
            fig.savefig(png_save_path)
        # plt.savefig("./res.png")
    fig.clear()
    axes[0].clear()
    axes[1].clear()
    plt.clf()
    plt.cla()
    plt.close()
    del arr


"""
Накладываем ли мы шум на всю клетку или на пиксель?
Можем ли мы получить зависимость значения уровня шума от частоты? Есть ли пороговое значение?
"""

if __name__ == '__main__':
    # N = 200
    # print("Generating dataset...")
    # for i in range(N):
    #     generate_example("../data/val_data", 100, 100, 2)
    #     if i % 10 == 0:
    #         print(f">> {i + 1} / {N}")

    basic_test("../data/val_data", 100, 100, 2, "../assets/pt/gcg2_model_3.2.pt", png=True, csv=False, txt=False)





