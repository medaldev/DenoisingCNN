import matplotlib.pyplot as plt
import torch

device = torch.device("cpu")

model = torch.load("../models/pt/uvych_tensor_denoiser_2_cpu_traced.pt").to(device)
model.eval()

# traced_script_module = torch.jit.script(model)
# traced_script_module.save("./uvych_tensor_denoiser_2_cpu_traced.pt")


from common.fstream import read_mc_tensor, read_matrix
import matplotlib.pyplot as pt
import os
import numpy as np

dir = "/home/amedvedev/projects/python/DenoisingCNN/data/datasets/gcg19/train/calculations/0a29a46e-cedb-4b67-8423-59bbb0b7a2bd/"

t = read_mc_tensor(os.path.join(dir, "Uvych2_abs_noised.tensor"))
t = torch.tensor(t, dtype=torch.float, device=device)
with torch.no_grad():
    res = model(t).squeeze().detach().cpu().numpy()
    print(res.shape)

uvych_real = read_matrix(os.path.join(dir, "Uvych2_abs.xls"))
uvych_diff = np.abs(res - uvych_real)
print(uvych_diff)

print(np.mean(uvych_diff))


fig, axes = plt.subplots(1, 3, figsize=(12, 3))

images = []

images.append(axes[0].imshow(res, cmap="jet"))
axes[0].set_title("Res")
images.append(axes[1].imshow(uvych_real, cmap="jet"))
axes[1].set_title("Real")
images.append(axes[2].imshow(uvych_diff, cmap="jet"))
axes[2].set_title("Diff")



for im in images:
    fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, format='%.7f')

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
fig.subplots_adjust(wspace=0.3, hspace=0.15)



# Save the full figure...
#fig.savefig(os.path.join(save_dir, f'{name}.png'))
plt.show(block=True)