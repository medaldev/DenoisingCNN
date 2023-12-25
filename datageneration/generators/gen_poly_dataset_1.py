from datetime import datetime

from datageneration.components.polygons import gen_data_polygons_1
from common.fstream import *
from common.matrix import add_noise


def generate_example(save_path, width, height, cell_size, csv=True, txt=True, png=True):
    surface = gen_data_polygons_1(width, height, cell_size, 5)
    # surface = gen_data_polygons_3(width, height, cell_size, 2, poly_radius=random.randint(4, 8))
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
            arrayToCsv(arr, os.path.join(save_path, "csv", "noised", f"{this_label}.csv"))
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
    N = 200
    print("Generating dataset...")
    for i in range(N):
        generate_example("../data/val_data", 100, 100, 2)
        if i % 10 == 0:
            print(f">> {i + 1} / {N}")
