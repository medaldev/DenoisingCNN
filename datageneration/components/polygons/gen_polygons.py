import random

from polygenerator import random_polygon

from common.value import renormalize
from datageneration.components.polygons.polygon import Polygon, generate_polygon
from datageneration.components.surface import Surface


def gen_data_polygons_1(width: int, height: int, cell_size: int, n_polygons: int, k0: float):
    surface = Surface(height, width, cell_size, k0)
    surface.set_k0(k0)
    x_step, y_step = width // n_polygons, height // n_polygons
    for i in range(n_polygons):
        i_rand, j_rand = random.randint(0, n_polygons - 1), random.randint(0, n_polygons - 1)
        k_min, k_max = 0.8, 1.5
        x_step_rand = random.randint(int(x_step * k_min), int(x_step * k_max))
        y_step_rand = random.randint(int(y_step * k_min), int(y_step * k_max))
        x_bounds = (i_rand * x_step, (i_rand + 1) * x_step_rand)
        y_bounds = (j_rand * y_step, (j_rand + 1) * y_step_rand)
        rescale_coords = lambda pair: (
        renormalize(pair[0], (0., 1.), x_bounds), renormalize(pair[1], (0., 1.), y_bounds))
        points = random_polygon(n_polygons)
        poly_data = list(map(rescale_coords, points))
        k_value = random.uniform(k0 * 1.1, k0 * 1.2)

        surface.add_figure(Polygon(poly_data, k=lambda x, y, kv=k_value: kv))
    return surface


def gen_data_polygons_2(width: int, height: int, cell_size: int, n_polygons: int, poly_radius: int, k0: float, k_min=0.7,
                        k_max=1.5):
    surface = Surface(height, width, cell_size, k0)

    x_step, y_step = width // n_polygons, height // n_polygons
    for i in range(n_polygons):
        i_rand, j_rand = random.randint(0, n_polygons - 1), random.randint(0, n_polygons - 1)
        x_bounds = (poly_radius + i_rand * x_step, (i_rand + 1) * x_step - poly_radius)
        y_bounds = (poly_radius + j_rand * y_step, (j_rand + 1) * y_step - poly_radius)
        poly_data = generate_polygon(
            center=(x_bounds[0] + (x_bounds[1] - x_bounds[0]) / 2, y_bounds[0] + (y_bounds[1] - y_bounds[0]) / 2),
            avg_radius=poly_radius * random.choice([k_min, k_max]),
            irregularity=0.35,
            spikiness=0.2,
            num_vertices=random.randint(5, 30))
        k_value = random.randint(1, 10)
        surface.add_figure(Polygon(poly_data, k=lambda x, y: k_value))
    return surface


def gen_data_polygons_3(width: int, height: int, cell_size: int, n_polygons: int, poly_radius: int, k0: float, k_min=0.7,
                        k_max=1.3):
    surface = Surface(height, width, cell_size, k0)
    surface.set_k0(k0)
    x_step, y_step = width // n_polygons, height // n_polygons
    for i in range(n_polygons):
        center = (random.randint(0, width), random.randint(0, height))
        poly_data = generate_polygon(center=center,
                                     avg_radius=poly_radius * random.choice([k_min, k_max]),
                                     irregularity=0.75,
                                     spikiness=0.4,
                                     num_vertices=20)
        k_value = random.uniform(k0 * 1.1, k0 * 1.2)
        surface.add_figure(Polygon(poly_data, k=lambda x, y, kv=k_value: kv))
    return surface
