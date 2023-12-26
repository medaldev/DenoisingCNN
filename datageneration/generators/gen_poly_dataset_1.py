from datetime import datetime

from datageneration.components.polygons import gen_data_polygons_1
from common.fstream import *
from common.matrix import add_noise
from datageneration.generators.abstractgenerator import AbstractGenerator
from common.stream import printProgressBar


class GenPolyDataset(AbstractGenerator):
    def generate_dataset(self, name, n, width, height, cell_size, pct_noise, csv=True, txt=True, png=True, parameters=None,
                         verbose=True):
        print(f"Generating dataset {name}...")
        for i in range(n):
            generate_example(f"../data/datasets/{name}", width, height, cell_size, pct_noise, csv, txt, png)
            # if verbose and i % 10 == 0:
            #     print(f">> {i + 1} / {n}")
            if verbose:
                printProgressBar(i + 1, n, prefix='Progress:', suffix='Complete', length=30)
        print()


def generate_example(save_path, width, height, cell_size, pct_noise, csv=True, txt=True, png=True):
    surface = gen_data_polygons_1(width, height, cell_size, 5)
    # surface = gen_data_polygons_3(width, height, cell_size, 2, poly_radius=random.randint(4, 8))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    arr = surface.matrix()
    if np.sum(arr) > 0:

        this_label = str(datetime.now())
        if csv:
            arrayToCsv(arr, os.path.join(save_path, "clear", "csv", f"{this_label}.csv"))
        if txt:
            array_save(arr, os.path.join(save_path, "clear", "txt", f"{this_label}.xls"))

        if png:
            axes[0].imshow(arr, interpolation='none')

        add_noise(arr, cell_size, pct=pct_noise)
        # print_matrix(arr)
        if csv:
            arrayToCsv(arr, os.path.join(save_path, "noised", "csv", f"{this_label}.csv"))
        if txt:
            array_save(arr, os.path.join(save_path, "noised", "txt", f"{this_label}.xls"))

        if png:
            axes[1].imshow(arr, interpolation='none')
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
    N = 1
    print("Generating dataset...")
    for i in range(N):
        generate_example("../data/val_data", 100, 100, 2, pct_noise=0.1)
        if i % 10 == 0:
            print(f">> {i + 1} / {N}")
