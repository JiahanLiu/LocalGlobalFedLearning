import torch

import os

def save_model_to_file(model, file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
    torch.save(model.state_dict(), file_path)

def load_model_from_file(model, file_path, device):
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint)