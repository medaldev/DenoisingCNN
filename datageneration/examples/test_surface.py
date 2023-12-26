from datageneration.components.prelude import *
from common.prelude import renormalize

if __name__ == '__main__':
    surface = Surface(300, 300, 1)
    surface\
        .add_figure(Circle(90, 90, 50))\
        .add_figure(Point(20, 20))\
        .add_figure(Rectangle(200, 200, 25, 50))\
        .add_figure(Triangle(25, 200, 100, 230, 70, 260))\
        .add_figure(Polygon([(170, 25), (180, 130), (200, 55), (250, 75), (220, 30), (240, 20)], k=lambda x, y: 1))\
        .add_figure(Polygon([(10, 25), (20, 50), (30, 25), (40, 50), (50, 25), (60, 50), (70, 25)], k=lambda x, y: 3))\
        .add_figure(Polygon(
            generate_polygon(center=(250, 250),
                             avg_radius=20,
                             irregularity=0.35,
                             spikiness=0.2,
                             num_vertices=16), k=lambda x, y: random.randint(1, 10)
        ))\
        .add_figure(Polygon(
            list(map(lambda pair: (renormalize(pair[0], (0., 1.), (110, 180)),
                                   renormalize(pair[1], (0., 1.), (150, 200)))
                     , random_polygon(10))), k=lambda x, y: 6
        ))
