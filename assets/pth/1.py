import torch

model = torch.load("./model_11.pth").cpu()

traced_script_module = torch.jit.script(model)

traced_script_module.save("traced_model_11.pt")
