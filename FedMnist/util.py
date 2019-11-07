# from model import nn_architectures, data_loader

import torch
import matplotlib.pyplot as plt

import os

def save_model_to_file(model, file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
    torch.save(model.state_dict(), file_path)

def load_model_from_file(model, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)

def print_shapes(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape) # torch.Size([1000, 1, 28, 28])

def plot_samples(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        