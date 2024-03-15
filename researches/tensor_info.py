import matplotlib




import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import gc
from multiprocessing import Pool

import json
#plt.style.use('dark_background')


def read_matrix(path):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(
            list(map(lambda row: list(map(lambda el: float(el), row.split())), file_matrix.read().strip().split("\n"))))
    return matrix

def read_mc_tensor(path):
    with open(path, "r", encoding="utf8") as file_tensor:
        data = json.load(file_tensor)
    return np.array(data)





dir = "/home/amedvedev/projects/python/DenoisingCNN/data/datasets/gcg19/train/calculations/0a29a46e-cedb-4b67-8423-59bbb0b7a2bd/"

example_path = dir #args[1]
fig, axes = plt.subplots(4, 16, figsize=(100, 50))

images = []

true_matrix = read_matrix(os.path.join(example_path, "Uvych2_abs.xls"))
tensor = read_mc_tensor(os.path.join(example_path, "Uvych2_re_noised.tensor"))


p = 0
for i in range(16):
    for j in range(4):
        images.append(axes[j, i].imshow(np.abs(true_matrix - tensor[p]), cmap="jet"))
        p+= 1


plt.tight_layout()

for row in axes:
    for ax in row:
        ax.set_xticks([])
        ax.set_yticks([])

for im in images:
    fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, format='%.7f')


fig.subplots_adjust(wspace=0.3, hspace=0.15)

# Save the full figure...
fig.savefig(os.path.join(".", 'uvych2_tensor.png'))
# plt.show()

# plt.clf()
# matplotlib.pyplot.close()

# del fig, axes, images
# gc.collect()

# plt.tight_layout()    # Your code here, the script continues to run
