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
import getopt
import json
import random
import sys

N_EPOCHS = 0
EPOCH_SATURATION = 0
MAX_EPOCHS = 0

def init():
    global N_EPOCHS
    global EPOCH_SATURATION
    global MAX_EPOCHS
    with open('config.json') as config_file:
        config = json.load(config_file)
        N_EPOCHS = int(config['machine_learning']['N_EPOCHS'])
        EPOCH_SATURATION = int(config['machine_learning']['EPOCH_SATURATION'])
        MAX_EPOCHS = int(config['machine_learning']['MAX_EPOCHS'])

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

    # net = network_architecture().to(device=DEVICE)
    # loss_fn = nn.NLLLoss()
    # optimizer = optim.Adam(net.parameters(), lr=FC_LEARNING_RATE)

    # init_weights(net)
    # for epoch in range(N_EPOCHS):
    #     losses = []
    #     for i in range(N_partitions): 
    #         (loss_i, local_param_i) = train_epoch(model=net, train_loader=train_loaders[i], loss_fn=loss_fn, optimizer=optimizer)
    #         losses.append(loss_i)
    #     acc = evaluate.accuracy(model=net, data_loader=test_loader)
    #     avg_loss = sum(losses) / len(losses)

    #     print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc) + " | Loss: " + str(avg_loss.item()))

def fed_learning(network_architecture, get_train_loader, get_test_loader, N_partitions, end_function):
    local_nets = [federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions, i) for i in range(N_partitions)]
    global_net = federated.Aggregated_Model(network_architecture, get_test_loader, N_partitions)

    avg_local_validation_accuracies = []
    n_epochs = 0
    end_condition = False
    while False == end_condition:
        local_params = []
        local_losses = []
        local_validation_accuracies = []
        local_accuracies = []
        for i in range(N_partitions): 
            local_nets[i].transfer_param_to_model(global_net.get_params())
            (loss_i, local_param_i) = local_nets[i].train()
            local_params.append(local_param_i)
            local_losses.append(loss_i)
            local_validation_accuracies.append(local_nets[i].get_validation_accuracy())
            local_accuracies.append(local_nets[i].get_accuracy())

        global_param = global_net.aggregate_central(local_params)

        avg_local_loss = sum(local_losses) / len(local_losses)
        avg_local_accuracy = sum(local_accuracies) / len(local_accuracies)
        avg_local_validation_accuracy = sum(local_validation_accuracies) / len(local_validation_accuracies)
        
        avg_local_validation_accuracies.append(avg_local_validation_accuracy)
        n_epochs = n_epochs + 1
        end_condition = end_function(n_epochs, avg_local_validation_accuracies)

        print("Epoch: " + str(n_epochs) + " | Accuracy: " + str(avg_local_accuracy) + " | Loss: " + str(avg_local_loss.item()))

    return (avg_local_accuracy, avg_local_loss.item(), n_epochs)

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

def stop_at_N_epochs_closure(N_epoch):
    def end_N_epochs(n_epoch, measures):
        if(n_epoch < N_epoch):
            return False
        else:
            return True

    return end_N_epochs

def stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION):
    def end_epoch_saturation(n_epoch, measures):
        if len(measures) > MAX_EPOCHS:
            return True

        if len(measures) <= EPOCH_SATURATION:
            return False

        highest_measure_index = 0
        highest_measure = measures[highest_measure_index]
        for i in range(len(measures)):
            if measures[i] > highest_measure:
                highest_measure_index = i
                highest_measure = measures[highest_measure_index]
        
        print("highest_measure_index: " + str(highest_measure_index) + " |len_losses: " + str(len(measures)) + " |highest_measure: " + str(highest_measure))
        if(highest_measure_index + EPOCH_SATURATION > len(measures)):
            return False
        else:
            return True
    
    return end_epoch_saturation

def main(): 
    init()
    
    stop_at_N_epochs = stop_at_N_epochs_closure(N_EPOCHS)
    stop_at_epoch_saturation = stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION)

    N_partitions = 3

    RESOLUTION = 2
    results_array = []
    results_array.append(['Semi-Balanced Percentage', 'Accuracy', 'Loss'])
    for i in range(RESOLUTION+1):
        balanced_percentage =  (100 / RESOLUTION) * i
        get_semibalanced_partitioned_train_loader = data_loader.get_semibalanced_partitioned_train_loaders_closure(balanced_percentage)   
        (accuracy, loss, N_epochs) = fed_learning(nn_architectures.NetFC_1, get_semibalanced_partitioned_train_loader, data_loader.get_unified_test_loader, N_partitions, stop_at_epoch_saturation) 
        results_array.append([balanced_percentage, accuracy, loss])

    results_path = 'results/demo_results.csv'
    with open(results_path, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in results_array:
            writer.writerow(row)

    # central_learning(nn_architectures.NetFC_1, data_loader.get_unified_train_loader, data_loader.get_unified_test_loader, N_EPOCHS)
    
    # fed_learning(nn_architectures.NetFC_1, data_loader.get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions)
    # local_learning(nn_architectures.NetFC_1, data_loader.get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions)
    
    # local_learning(nn_architectures.NetFC_1, data_loader.get_unbalanced_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions)
    # fed_learning(nn_architectures.NetFC_1, data_loader.get_unbalanced_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions)
    # local_learning(nn_architectures.NetFC_1, data_loader.get_unbalanced_partitioned_train_loaders, data_loader.get_unbalanced_partitioned_test_loaders, N_EPOCHS, N_partitions)

    # get_semibalanced_partitioned_train_loader_fifty_percent = data_loader.get_semibalanced_partitioned_train_loaders_closure()   
    # fed_learning(nn_architectures.NetFC_1, get_semibalanced_partitioned_train_loader_fifty_percent, data_loader.get_unified_test_loader, N_EPOCHS, N_partitions) 

if __name__ == "__main__":
    main()


