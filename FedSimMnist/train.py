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

def transfer_param_to_model(model, param):
    for w_model, w_param in zip(model.parameters(), param):
        w_model.data = w_param.data

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
    N_partitions = len(local_params)
    for w_global in global_model.parameters():
        w_global.data = w_global.data * 0
    for local_param in local_params:
        for w_global, w_local in zip(global_model.parameters(), local_param):
                w_global.data += w_local.data / N_partitions
    return global_model.parameters()

def central_learning(network_architecture, get_train_loader, get_test_loader):
    (train_loader, validation_loader) = get_train_loader()
    test_loader = get_test_loader(1)

    net = network_architecture().to(device=DEVICE)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=FC_LEARNING_RATE)

    init_weights(net)

    for epoch in range(N_EPOCHS):
        (loss, params) = train_epoch(model=net, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
        acc = evaluate.accuracy(model=net, data_loader=test_loader)
        loss = evaluate.loss(model=net, data_loader=train_loader, loss_fn=loss_fn)

        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc) + " | Loss: " + str(loss.item()))

def central_learning_split_sets(network_architecture, get_unified_loader, get_partitioned_loader, get_test_loader, N_partitions):
    (train_loader_all, not_used) = get_unified_loader()
    (train_loaders, validation_loader) = get_partitioned_loader(N_partitions)
    test_loader = get_test_loader(N_partitions)

    net = network_architecture().to(device=DEVICE)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=FC_LEARNING_RATE)

    init_weights(net)
    for epoch in range(N_EPOCHS):
        loss = None
        local_param = None
        for i in range(N_partitions): 
            (loss, local_param) = train_epoch(model=net, train_loader=train_loaders[i], loss_fn=loss_fn, optimizer=optimizer)
        acc = evaluate.accuracy(model=net, data_loader=test_loader)
        loss = evaluate.loss(model=net, data_loader=train_loader_all, loss_fn=loss_fn)

        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc) + " | Loss: " + str(loss.item()))

def fed_learning(network_architecture, get_train_loader, get_test_loader, N_partitions):
    (train_loaders, validation_loader) = get_train_loader(N_partitions)
    test_loader = get_test_loader(N_partitions)

    local_nets = [network_architecture().to(device=DEVICE) for i in range(N_partitions)]
    global_net = network_architecture().to(device=DEVICE)
    loss_fn = nn.NLLLoss()
    optimizers = [optim.Adam(local_nets[i].parameters(), lr=FC_LEARNING_RATE) for i in range(N_partitions)]

    init_weights(global_net)
    for epoch in range(N_EPOCHS):
        local_losses = []
        local_params = []
        for i in range(N_partitions): 
            transfer_param_to_model(model=local_nets[i], param=global_net.parameters())
            (loss_i, local_param_i) = train_epoch(model=local_nets[i], train_loader=train_loaders[i], loss_fn=loss_fn, optimizer=optimizers[i])
            local_losses.append(loss_i)
            local_params.append(local_param_i)

        global_param = aggregate_central(global_model=global_net, local_params=local_params)

        acc = evaluate.accuracy(model=global_net, data_loader=test_loader)
        global_loss = evaluate.loss(model=local_nets[0], data_loader=train_loaders[0], loss_fn=loss_fn)

        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc) + " | L_Loss: " + str(local_losses[0].item()) + " | G_Loss: " + str(global_loss.item()))

def local_learning(network_architecture, get_train_loader, get_test_loader, N_partitions):
    (train_loaders, validation_loader) = get_train_loader(N_partitions)
    test_loader = get_test_loader(N_partitions)

    local_nets = [network_architecture().to(device=DEVICE) for i in range(N_partitions)]
    loss_fn = nn.NLLLoss()
    optimizers = [optim.Adam(local_nets[i].parameters(), lr=FC_LEARNING_RATE) for i in range(N_partitions)]

    for i in range(N_partitions):
        init_weights(local_nets[i])

    for epoch in range(N_EPOCHS):
        local_losses = []
        local_accuracies = []
        acc = -1
        for i in range(N_partitions): 
            (loss_i, local_param_i) = train_epoch(model=local_nets[i], train_loader=train_loaders[i], loss_fn=loss_fn, optimizer=optimizers[i])
            if(len(test_loader) == 1):
                acc = evaluate.accuracy(model=local_nets[i], data_loader=test_loader)
            else:
                acc = evaluate.accuracy(model=local_nets[i], data_loader=test_loader[i])
            local_losses.append(loss_i)
            local_accuracies.append(acc)

        avg_local_accuracy = sum(local_accuracies) / len(local_accuracies)
        avg_local_loss = sum(local_losses) / len(local_losses)

        print("Epoch: " + str(epoch) + " | Avg_L_Accuracy: " + str(avg_local_accuracy) + " | Avg_L_Loss: " + str(avg_local_loss.item()))

def main(): 
    central_learning(network_architecture=nn_architectures.NetFC, get_train_loader=data_loader.get_unified_train_loader, 
        get_test_loader=data_loader.get_unified_test_loader)
    # central_learning_split_sets(network_architecture=nn_architectures.NetFC, get_unified_loader=data_loader.get_unified_train_loader, 
    #     get_partitioned_loader=data_loader.get_random_partitioned_train_loaders, get_test_loader=data_loader.get_unified_test_loader, N_partitions=3)
    
    # fed_learning(network_architecture=nn_architectures.NetFC, get_train_loader=data_loader.get_random_partitioned_train_loaders, 
    #     get_test_loader=data_loader.get_unified_test_loader, N_partitions=3)
    # fed_learning(network_architecture=nn_architectures.NetFC, get_train_loader=data_loader.get_unbalanced_partitioned_train_loaders, 
    #     get_test_loader=data_loader.get_unified_test_loader, N_partitions=3)

    # local_learning(network_architecture=nn_architectures.NetFC, get_train_loader=data_loader.get_random_partitioned_train_loaders, 
    #     get_test_loader=data_loader.get_unified_test_loader, N_partitions=3)
    # local_learning(network_architecture=nn_architectures.NetFC, get_train_loader=data_loader.get_unbalanced_partitioned_train_loaders, 
    #     get_test_loader=data_loader.get_unified_test_loader, N_partitions=3)
    # local_learning(network_architecture=nn_architectures.NetFC, get_train_loader=data_loader.get_unbalanced_partitioned_train_loaders, 
    #     get_test_loader=data_loader.get_unbalanced_partitioned_test_loaders, N_partitions=3)

if __name__ == "__main__":
    main()


