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

def central_learning(network_architecture, get_train_loader, get_test_loader, end_function, learning_rate):
    net = federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions=0, node_id=0, learning_rate=learning_rate)

    end_condition = False
    epoch_n = 0
    validation_accuracies = []
    while False == end_condition:
        (loss_i, local_param_i) = net.train()
        loss = net.get_loss()
        validation_accuracy = net.get_validation_accuracy()
        acc = net.get_accuracy()

        validation_accuracies.append(validation_accuracy)
        epoch_n = epoch_n + 1
        end_condition, optimal_epoch = end_function(epoch_n, validation_accuracies)

        print("Epoch: " + str(epoch_n) + " | Accuracy: " + str(acc) + " | Loss: " + str(loss.item()))
    
    return optimal_epoch, validation_accuracies[optimal_epoch]

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

def fed_learning(network_architecture, get_train_loader, get_test_loader, N_partitions, end_function, learning_rate):
    local_nets = [federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions, node_i, learning_rate) for node_i in range(N_partitions)]
    global_net = federated.Aggregated_Model(network_architecture, get_test_loader, N_partitions)

    end_condition = False
    epoch_n = 0
    avg_local_validation_accuracies = []
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
        avg_local_validation_accuracy = sum(local_validation_accuracies) / len(local_validation_accuracies)
        avg_local_accuracy = sum(local_accuracies) / len(local_accuracies)
        
        avg_local_validation_accuracies.append(avg_local_validation_accuracy)
        epoch_n = epoch_n + 1
        end_condition = end_function(epoch_n, avg_local_validation_accuracies)

        print("Epoch: " + str(epoch_n) + " | Accuracy: " + str(avg_local_accuracy) + " | Loss: " + str(avg_local_loss.item()))

    return (avg_local_accuracy, avg_local_loss.item(), epoch_n)

def local_learning(network_architecture, get_train_loader, get_test_loader, N_partitions, end_function, learning_rate):
    local_nets = [federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions, node_i, learning_rate) for node_i in range(N_partitions)]

    end_condition = False
    epoch_n = 0
    avg_local_validation_accuracies = []
    while False == end_condition:
        local_losses = []
        local_validation_accuracies = []
        local_accuracies = []
        for i in range(len(local_nets)):
            (loss_i, local_param_i) = local_nets[i].train()
            local_losses.append(loss_i)
            local_validation_accuracies.append(local_nets[i].get_validation_accuracy())
            local_accuracies.append(local_nets[i].get_accuracy())

        avg_local_loss = sum(local_losses) / len(local_losses)
        avg_local_validation_accuracy = sum(local_validation_accuracies) / len(local_validation_accuracies)
        avg_local_accuracy = sum(local_accuracies) / len(local_accuracies)

        avg_local_validation_accuracies.append(avg_local_validation_accuracy)
        epoch_n = epoch_n + 1
        end_condition = end_function(epoch_n, avg_local_validation_accuracies)

        print("Epoch: " + str(epoch_n) + " | Avg_L_Accuracy: " + str(avg_local_accuracy) + " | Avg_L_Loss: " + str(avg_local_loss.item()))

def stop_at_N_epochs_closure(N_epoch):
    def end_N_epochs(n_epoch, measures):
        if(n_epoch < N_epoch):
            return False, N_epoch
        else:
            return True, N_epoch

    return end_N_epochs

def stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION):
    def end_epoch_saturation(n_epoch, measures):
        highest_measure_index = 0
        highest_measure = measures[highest_measure_index]
        for i in range(len(measures)):
            if measures[i] > highest_measure:
                highest_measure_index = i
                highest_measure = measures[highest_measure_index]
        
        if len(measures) > MAX_EPOCHS:
            return True, highest_measure_index

        if len(measures) <= EPOCH_SATURATION:
            return False, highest_measure_index

        if(highest_measure_index + EPOCH_SATURATION > len(measures)):
            return False, highest_measure_index
        else:
            return True, highest_measure_index
    
    return end_epoch_saturation

def main(): 
    with open('config.json') as config_file:
        config = json.load(config_file)
        N_PARTITIONS = int(config['machine_learning']['N_PARTITIONS'])
        N_EPOCHS = int(config['machine_learning']['N_EPOCHS'])
        EPOCH_SATURATION = int(config['machine_learning']['EPOCH_SATURATION'])
        MAX_EPOCHS = int(config['machine_learning']['MAX_EPOCHS'])
        FC_LEARNING_RATE = float(config['machine_learning']['FC_LEARNING_RATE'])
    
    stop_at_N_epochs = stop_at_N_epochs_closure(N_EPOCHS)
    stop_at_epoch_saturation = stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION)

    # RESOLUTION = 2
    # results_array = []
    # results_array.append(['Semi-Balanced Percentage', 'Accuracy', 'Loss'])
    # for i in range(RESOLUTION+1):
    #     balanced_percentage =  (100 / RESOLUTION) * i
    #     get_semibalanced_partitioned_train_loader = data_loader.get_semibalanced_partitioned_train_loaders_closure(balanced_percentage, batch_size=256)   
    #     (accuracy, loss, N_epochs) = fed_learning(nn_architectures.NetFC_1, get_semibalanced_partitioned_train_loader, data_loader.get_unified_test_loader, N_partitions, stop_at_epoch_saturation, FC_LEARNING_RATE) 
    #     results_array.append([balanced_percentage, accuracy, loss])

    # results_path = 'results/demo_results.csv'
    # with open(results_path, mode='w') as csv_file:
    #     writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for row in results_array:
    #         writer.writerow(row)

    # get_unified_train_loader = data_loader.get_unified_train_loader_closure(batch_size=256)
    # central_learning(nn_architectures.NetFC_1, get_unified_train_loader, data_loader.get_unified_test_loader, stop_at_N_epochs, FC_LEARNING_RATE)
    
    # get_random_partitioned_train_loaders = data_loader.get_random_partitioned_train_loaders_closure(batch_size=256)
    # fed_learning(nn_architectures.NetFC_1, get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)
    # local_learning(nn_architectures.NetFC_1, get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)
    
    # get_unbalanced_partitioned_train_loaders = data_loader.get_unbalanced_partitioned_train_loaders_closure(batch_size=256)
    # local_learning(nn_architectures.NetFC_1, get_unbalanced_partitioned_train_loaders, data_loader.get_unified_test_loader, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)
    # fed_learning(nn_architectures.NetFC_1, get_unbalanced_partitioned_train_loaders, data_loader.get_unified_test_loader, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)
    # local_learning(nn_architectures.NetFC_1, get_unbalanced_partitioned_train_loaders, data_loader.get_unbalanced_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)

    # get_semibalanced_partitioned_train_loader_fifty_percent = data_loader.get_semibalanced_partitioned_train_loaders_closure(50, batch_size=256)   
    # fed_learning(nn_architectures.NetFC_1, get_semibalanced_partitioned_train_loader_fifty_percent, data_loader.get_unified_test_loader, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE) 

if __name__ == "__main__":
    main()


