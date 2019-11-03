import evaluate
import util
from model import nn_architectures, data_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import random
    
print("Torch Version: " + torch.__version__)
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using Cuda")

torch.manual_seed(random.random() * 100)

N_EPOCHS = 30
LOG_INTERVAL = 10
FC_LEARNING_RATE = 0.001

def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0.01)

def transfer_params_to_model(model, params):
    for w_model, w_params in zip(model.parameters(), params):
        w_model.data = w_params.data

def train_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        train_x = train_x.to(DEVICE)
        train_y = train_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = loss_fn(outputs, train_y)
        loss.backward()
        optimizer.step()

    return (loss, model.parameters())

def aggregate_central(global_model, local_params):
    for w_global, w_local in zip(global_model.parameters(), local_params):
            w_global.data = w_local.data
    return global_model.parameters()

def fed_learning():
    train_loader, validation_loader, test_loader = data_loader.get_loaders()

    local_net = nn_architectures.NetFC().to(device=DEVICE)
    global_net = nn_architectures.NetFC().to(device=DEVICE)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(local_net.parameters(), lr=FC_LEARNING_RATE)

    init_weights(global_net)
    for epoch in range(N_EPOCHS):
        transfer_params_to_model(model=local_net, params=global_net.parameters())

        (local_loss, local_params) = train_epoch(model=local_net, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
        global_params = aggregate_central(global_model=global_net, local_params=local_params)

        acc = evaluate.accuracy(model=global_net, data_loader=test_loader)
        global_loss = evaluate.loss(model=local_net, data_loader=train_loader, loss_fn=loss_fn)

        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc) + " | L_Loss: " + str(local_loss.item()) + " | G_Loss: " + str(global_loss.item()))

def main(): 
    fed_learning()

if __name__ == "__main__":
    main()


