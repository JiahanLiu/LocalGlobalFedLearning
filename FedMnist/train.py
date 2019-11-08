import evaluate
import federated
import util
from model import nn_architectures, data_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import csv
import json
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

def central_learning_split_interweaved(network_architecture, get_unified_loader, get_partitioned_loader, get_test_loader, N_partitions):
    (train_loaders, validation_loader) = get_partitioned_loader(N_partitions)
    test_loader = get_test_loader(N_partitions)

    net = network_architecture().to(device=DEVICE)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=FC_LEARNING_RATE)

    init_weights(net)
    for epoch in range(N_EPOCHS):
        losses = []
        for i in range(N_partitions): 
            (loss_i, local_param_i) = train_epoch(model=net, train_loader=train_loaders[i], loss_fn=loss_fn, optimizer=optimizer)
            losses.append(loss_i)
        acc = evaluate.accuracy(model=net, data_loader=test_loader)
        avg_loss = sum(losses) / len(losses)

        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc) + " | Loss: " + str(avg_loss.item()))

def fed_learning(network_architecture, get_train_loader, get_test_loader, n_epochs, N_partitions):
    local_nets = [federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions, i) for i in range(N_partitions)]
    global_net = federated.Aggregated_Model(network_architecture, get_test_loader, N_partitions)

    for epoch in range(n_epochs):
        local_params = []
        local_losses = []
        local_accuracies = []
        for i in range(N_partitions): 
            local_nets[i].transfer_param_to_model(global_net.get_params())
            (loss_i, local_param_i) = local_nets[i].train()
            local_params.append(local_param_i)
            local_losses.append(loss_i)
            local_accuracies.append(local_nets[i].get_accuracy())

        global_param = global_net.aggregate_central(local_params)

        avg_local_accuracy = sum(local_accuracies) / len(local_accuracies)
        avg_local_loss = sum(local_losses) / len(local_losses)

        print("Epoch: " + str(epoch) + " | Accuracy: " + str(avg_local_accuracy) + " | Loss: " + str(avg_local_loss.item()))

    return (avg_local_accuracy, avg_local_loss.item())

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
    with open('config.json') as config_file:
        config = json.load(config_file)
        N_EPOCHS = int(config['machine_learning']['N_EPOCHS'])
    
    N_partitions = 3

    with open('demo_results.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Semi-Balanced Percentage', 'Accuracy', 'Loss'])
        for i in range(10):
            get_semibalanced_partitioned_train_loader_fifty_percent = data_loader.get_semibalanced_partitioned_train_loaders_closure(10 * i)   
            (accuracy, loss) = fed_learning(nn_architectures.NetFC, get_semibalanced_partitioned_train_loader_fifty_percent, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions) 
            writer.writerow([10 * i, accuracy, loss])

    # central_learning(nn_architectures.NetFC, data_loader.get_unified_train_loader, data_loader.get_unified_test_loader, N_EPOCHS)
    
    # fed_learning(nn_architectures.NetFC, data_loader.get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions)
    # local_learning(nn_architectures.NetFC, data_loader.get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions)
    
    # local_learning(nn_architectures.NetFC, data_loader.get_unbalanced_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions)
    # fed_lear  ning(nn_architectures.NetFC, data_loader.get_unbalanced_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions)
    # local_learning(nn_architectures.NetFC, data_loader.get_unbalanced_partitioned_train_loaders, data_loader.get_unbalanced_partitioned_test_loaders, N_EPOCHS, N_partitions)

    # get_semibalanced_partitioned_train_loader_fifty_percent = data_loader.get_semibalanced_partitioned_train_loaders_closure(50)   
    # fed_learning(nn_architectures.NetFC, get_semibalanced_partitioned_train_loader_fifty_percent, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions) 

if __name__ == "__main__":
    main()


