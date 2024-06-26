import matplotlib.pyplot as plt
import torch

def model_serialize(model_path, path_save, device='cpu'):

    device = torch.device(device)

    model = torch.load(model_path).to(device)
    model.eval()

    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path_save)


model_serialize("/home/amedvedev/projects/python/DenoisingCNN/assets/pt/J_matrix_denoiser_30pct_v_1.pt",
                '/home/amedvedev/projects/rust/gcg_rs/models/J_matrix_denoiser_30pct_v_1.pt')