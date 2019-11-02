import evaluate
import util
from model import nn_architectures, data_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import os
import random

print("Torch Version: " + torch.__version__)
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using Cuda")
    
DIRPATH = os.getcwd()
MODELPATH = DIRPATH + '/model/training_in_progress.pkl'

N_EPOCHS = 30
LOG_INTERVAL = 10
FC_LEARNING_RATE = 0.001

torch.manual_seed(random.random() * 100)

train_loader, validation_loader, test_loader = data_loader.get_loaders()

local_net = nn_architectures.NetFC().to(device=DEVICE)
global_net = nn_architectures.NetFC().to(device=DEVICE)

loss_fn = nn.NLLLoss()
optimizer = optim.Adam(local_net.parameters(), lr=FC_LEARNING_RATE)

def transfer_global_params_to_local(local_model, global_params):
    for w_global, w_local in zip(global_params, local_model.parameters()):
        w_local.data = w_global.data

def train_epoch_local(local_model, loss_fn, optimizer):
    local_model.train()
    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        train_x = train_x.to(DEVICE)
        train_y = train_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = local_model(train_x)
        loss = loss_fn(outputs, train_y)
        loss.backward()
        # for f in local_model.parameters():
        #     f.grad.data = f.grad.data
        optimizer.step()

    return (loss, local_model.parameters())

def aggregate_central(global_model, local_params):
    for w_global, w_local in zip(global_model.parameters(), local_params):
            w_global.data = w_local.data
    return global_model.parameters()

def save_model_global_var():
    global local_net
    torch.save(local_net, MODELPATH)

def load_model_global_var():
    global local_net
    local_net = torch.load(MODELPATH)

def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0.01)

def fed_learning():
    init_weights(global_net)
    for epoch in range(N_EPOCHS):

        transfer_global_params_to_local(
            local_model=local_net, 
            global_params=global_net.parameters())

        (local_loss, local_params) = train_epoch_local(
            local_model=local_net,
            loss_fn=loss_fn,
            optimizer=optimizer)

        global_params = aggregate_central(
            global_model=global_net, 
            local_params=local_params)

        # save_model_global_var()
        # load_model_global_var()
        acc = evaluate.evaluate(
            model=global_net, 
            test_loader=test_loader)

        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc))

fed_learning()

# (train_loaders, test_loader) = data_loader.get_random_partitioned_loaders(N_partitions=3)

# print(len(train_loaders))
