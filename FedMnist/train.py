import evaluate
import federated
import util
from model import nn_architectures, data_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import configparser
import random
    
print("Torch Version: " + torch.__version__)
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using Cuda")
torch.manual_seed(random.random() * 100)

def central_learning(network_architecture, get_train_loader, get_test_loader, n_epochs):
    net = federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions=0, node_id=0)

    for epoch in range(n_epochs):
        (loss_i, local_param_i) = net.train()
        acc = net.get_accuracy()
        loss = net.get_loss()

        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc) + " | Loss: " + str(loss.item()))

def fed_learning(network_architecture, get_train_loader, get_test_loader, n_epochs, N_partitions):
    local_nets = [federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions, i) for i in range(N_partitions)]
    global_net = federated.Aggregated_Model(network_architecture, get_test_loader, N_partitions)

    for epoch in range(n_epochs):
        local_losses = []
        local_params = []
        for i in range(N_partitions): 
            local_nets[i].transfer_param_to_model(global_net.get_params())
            (loss_i, local_param_i) = local_nets[i].train()
            local_losses.append(loss_i)
            local_params.append(local_param_i)

        global_param = global_net.aggregate_central(local_params)

        acc = global_net.get_accuracy()
        # global_loss = evaluate.loss(model=local_nets[0], data_loader=train_loaders[0], loss_fn=loss_fn)

        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc))

def local_learning(network_architecture, get_train_loader, get_test_loader, n_epochs, N_partitions):
    local_nets = [federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions, i) for i in range(N_partitions)]

    for epoch in range(n_epochs):
        local_losses = []
        local_accuracies = []
        acc = -1
        for i in range(len(local_nets)):
            (loss_i, local_param_i) = local_nets[i].train()
            acc = local_nets[i].get_accuracy()
            local_losses.append(loss_i)
            local_accuracies.append(acc)

        avg_local_accuracy = sum(local_accuracies) / len(local_accuracies)
        avg_local_loss = sum(local_losses) / len(local_losses)

        print("Epoch: " + str(epoch) + " | Avg_L_Accuracy: " + str(avg_local_accuracy) + " | Avg_L_Loss: " + str(avg_local_loss.item()))

def main(): 
    config = configparser.RawConfigParser()
    config.read('config.cfg')
    N_EPOCHS = int(config['DEFAULT']['N_EPOCHS'])

    # get_semibalanced_partitioned_train_loader_fifty_percent = data_loader.get_semibalanced_partitioned_train_loaders_closure(50)
    
    # central_learning(nn_architectures.NetFC, data_loader.get_unified_train_loader, data_loader.get_unified_test_loader, N_EPOCHS)
    
    # fed_learning(nn_architectures.NetFC, data_loader.get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions=3)
    # fed_learning(nn_architectures.NetFC, data_loader.get_unbalanced_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions=3)
    # fed_learning(nn_architectures.NetFC, get_semibalanced_partitioned_train_loader_fifty_percent, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions=3)

    local_learning(nn_architectures.NetFC, data_loader.get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions=3)
    # local_learning(nn_architectures.NetFC, data_loader.get_unbalanced_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions=3)
    # local_learning(nn_architectures.NetFC, data_loader.get_unbalanced_partitioned_train_loaders, data_loader.get_unbalanced_partitioned_test_loaders, N_EPOCHS, N_partitions=3)

if __name__ == "__main__":
    main()


