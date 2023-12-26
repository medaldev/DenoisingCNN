from datageneration.generators.gen_poly_dataset_1 import GenPolyDataset


class AbstractGenerator:

    def generate_dataset(self, name, n, width, height, cell_size, csv=True, txt=True, png=True, parameters=None, verbose=True):
        pass

def get_basic_generator():
    return GenPolyDataset()