
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def read_matrix(path, coeff_norm=1.0):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(
            list(map(lambda row: list(map(lambda el: float(el), row.split())),
                     file_matrix.read().strip().split("\n")))) / coeff_norm
    return matrix

def load2d(path, pct=0.01):
    uvych = read_matrix(path)
    uvych_noised = uvych.copy()

    for i in range(len(uvych)):
        for j in range(len(uvych[i])):
            ppp = np.random.uniform(low=0, high=pct)
            uvych_noised[i][j] *= (1 + ppp)

    return uvych, uvych_noised



path = input("Enter the path: ")
uvych, uvych_noised = load2d(path, 0.0001)
# Calculate a 2D power spectrum
psd2D = np.log(fftpack.fftshift( fftpack.fft2(uvych) ))
psd2D_noised = np.log(fftpack.fftshift( fftpack.fft2(uvych_noised) ))
fig, axes = plt.subplots(1, 3, figsize=(15, 7))
i1 = axes[0].imshow( psd2D.real, cmap="jet")
i2 = axes[1].imshow(psd2D_noised.real, cmap="jet")
i3 = axes[2].imshow(((  psd2D  -  psd2D_noised ).imag), cmap="jet")
plt.colorbar(i1, orientation='vertical', fraction=0.046, pad=0.04)
plt.colorbar(i2, orientation='vertical', fraction=0.046, pad=0.04)
plt.colorbar(i3, orientation='vertical', fraction=0.046, pad=0.04)
plt.show()


path_save = "result.xls"
with open(path_save, "w") as file:
    print(psd2D.tolist(), file=file)