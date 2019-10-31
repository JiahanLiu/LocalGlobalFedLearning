import nn_architectures

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import os
import random

print("Torch Version: " + torch.__version__)

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using Cuda")
    
DIRPATH = os.getcwd()
DATAPATH = DIRPATH + '/data/'
MODELPATH = DIRPATH + '/model/training_in_progress.pkl'

BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 1
N_EPOCHS = 30
LOG_INTERVAL = 10
FC_LEARNING_RATE = 0.001

torch.manual_seed(random.random() * 100)

train_dataset = datasets.MNIST(
    root=DATAPATH, 
    train=True, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]), 
    download=True)
test_dataset = datasets.MNIST(
    root=DATAPATH, 
    train=False, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]),
    download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

local_net = nn_architectures.NetFC().to(device=DEVICE)
global_net = nn_architectures.NetFC().to(device=DEVICE)

loss_fn = nn.NLLLoss()
optimizer = optim.Adam(local_net.parameters(), lr=FC_LEARNING_RATE)
test_set = enumerate(test_loader)

def train_epoch_local(local_model, global_params, loss_fn, optimizer, first_pass):
    local_model.train()
    if first_pass == False:
        for w_global, w_local in zip(global_params, local_model.parameters()):
            w_local.data = w_global.data
    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        train_x = train_x.to(DEVICE)
        train_y = train_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = local_model(train_x)
        loss = loss_fn(outputs, train_y)
        loss.backward()
        for f in local_model.parameters():
            f.grad.data = f.grad.data
        optimizer.step()

    return (loss, local_model.parameters())

def aggregate_central(global_model, local_params):
    for w_global, w_local in zip(global_model.parameters(), local_params):
            w_global.data = w_local.data
    return (loss, global_model.parameters())

def save_model_global_var():
    global local_net
    torch.save(local_net, MODELPATH)

def load_model_global_var():
    global local_net
    local_net = torch.load(MODELPATH)

def evaluate(model):
    acc = -1
    correct = 0
    model.eval()
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(DEVICE)
            test_y = test_y.to(DEVICE)
            output = model(test_x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(test_y.view_as(pred)).sum().item()
        acc = (100 * correct) / len(test_loader.dataset)

    return acc

def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0.01)

for epoch in range(N_EPOCHS):
    init_weights(local_net)
    first_pass = False
    if epoch == 0:
        first_pass = True
    (loss, local_params) = train_epoch_local(local_net, local_params,loss_fn, optimizer, first_pass)
    (loss, global_params) = aggregate_central(global_net, local_params)
    # save_model_global_var()
    # load_model_global_var()
    acc = evaluate(global_net)
    print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc) + " | Loss: " + str(loss.item()))

def print_shapes():
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape) # torch.Size([1000, 1, 28, 28])

def plot_samples():
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
        