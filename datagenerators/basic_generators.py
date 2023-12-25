import numpy as np
import torch

from datagenerators.components.polygon import generate_polygon
from polygenerator import random_polygon
import random
from datetime import datetime
from PIL import Image
import os






















def renormalize(n, from_range, to_range):
    delta1 = from_range[1] - from_range[0]
    delta2 = to_range[1] - to_range[0]
    return (delta2 * (n - from_range[0]) / delta1) + to_range[0]


def rescale_val(val, min, max):
  return (val - min) / (max - min)

def arrayToImage(matrix: np.ndarray, save_path="./res.png"):
    im = Image.fromarray(matrix).convert("RGB")
    im.save(save_path)


def array_save(matrix: np.ndarray, save_path: str):
    h, w = matrix.shape
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf8") as file:
        for i in range(h):
            for j in range(w):
                print(matrix[i][j], file=file, end=" ")
            print(file=file)


def arrayToCsv(matrix: np.ndarray, save_path: str):
    h, w = matrix.shape
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf8") as csv_file:
        for i in range(h):
            for j in range(w):
                print(rescale_val(i, 0, h), rescale_val(j, 0, w), matrix[i][j], file=csv_file, end=", ")


def gen_data_polygons_1(width: int, height: int, cell_size: int, n_polygons: int):
    surface = Surface(height, width, cell_size)
    x_step, y_step = width // n_polygons, height // n_polygons
    for i in range(n_polygons):
        i_rand, j_rand = random.randint(0, n_polygons - 1), random.randint(0, n_polygons - 1)
        k_min, k_max = 0.8, 1.5
        x_step_rand = random.randint(int(x_step * k_min), int(x_step * k_max))
        y_step_rand = random.randint(int(y_step * k_min), int(y_step * k_max))
        x_bounds = (i_rand * x_step, (i_rand + 1) * x_step_rand)
        y_bounds = (j_rand * y_step, (j_rand + 1) * y_step_rand)
        rescale_coords = lambda pair: (renormalize(pair[0], (0., 1.), x_bounds), renormalize(pair[1], (0., 1.), y_bounds))
        points = random_polygon(n_polygons)
        poly_data = list(map(rescale_coords, points))
        surface.add_figure(Polygon(poly_data))
    return surface


def gen_data_polygons_2(width: int, height: int, cell_size: int, n_polygons: int, poly_radius: int, k_min=0.7, k_max=1.5):
    surface = Surface(height, width, cell_size)
    x_step, y_step = width // n_polygons, height // n_polygons
    for i in range(n_polygons):
        i_rand, j_rand = random.randint(0, n_polygons - 1), random.randint(0, n_polygons - 1)
        x_bounds = (poly_radius + i_rand * x_step, (i_rand + 1) * x_step - poly_radius)
        y_bounds = (poly_radius + j_rand * y_step, (j_rand + 1) * y_step - poly_radius)
        poly_data = generate_polygon(center=(x_bounds[0] + (x_bounds[1] - x_bounds[0]) / 2, y_bounds[0] + (y_bounds[1] - y_bounds[0]) / 2),
                             avg_radius=poly_radius * random.choice([k_min, k_max]),
                             irregularity=0.35,
                             spikiness=0.2,
                             num_vertices=random.randint(5, 30))
        surface.add_figure(Polygon(poly_data))
    return surface


def gen_data_polygons_3(width: int, height: int, cell_size: int, n_polygons: int, poly_radius: int, k_min=0.7, k_max=1.3):
    surface = Surface(height, width, cell_size)
    x_step, y_step = width // n_polygons, height // n_polygons
    for i in range(n_polygons):
        center = (random.randint(0, width), random.randint(0, height))
        poly_data = generate_polygon(center=center,
                             avg_radius=poly_radius * random.choice([k_min, k_max]),
                             irregularity=0.75,
                             spikiness=0.4,
                             num_vertices=20)
        surface.add_figure(Polygon(poly_data))
    return surface


def rescale_array(a, to=(0, +1)):
  return np.interp(a, (a.min(), a.max()), to)
def read_matrix(path):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(list(map(lambda row: list(map(lambda el: float(el) , row.split())), file_matrix.read().strip().split("\n"))))
    return matrix

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



# if __name__ == '__main__':
#     surface = Surface(300, 300, 1)
#     surface\
#         .add_figure(Circle(90, 90, 50))\
#         .add_figure(Point(20, 20))\
#         .add_figure(Rectangle(200, 200, 25, 50))\
#         .add_figure(Triangle(25, 200, 100, 230, 70, 260))\
#         .add_figure(Polygon([(170, 25), (180, 130), (200, 55), (250, 75), (220, 30), (240, 20)]))\
#         .add_figure(Polygon([(10, 25), (20, 50), (30, 25), (40, 50), (50, 25), (60, 50), (70, 25)]))\
#         .add_figure(Polygon(
#             generate_polygon(center=(250, 250),
#                              avg_radius=20,
#                              irregularity=0.35,
#                              spikiness=0.2,
#                              num_vertices=16)
#         ))\
#         .add_figure(Polygon(
#             list(map(lambda pair: (renormalize(pair[0], (0., 1.), (110, 180)),
#                                    renormalize(pair[1], (0., 1.), (150, 200)))
#                      , random_polygon(10))),
#         ))

