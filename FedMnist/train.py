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
import sys

DEBUG = True

def print_results(epoch_n, loss, validation_accuracy, acc):
    if DEBUG:
        print("Epoch: " + epoch_n + " | Loss: " + loss + " | ValAcc: " + validation_accuracy + " | Acc: " + acc)

def central_learning(network_architecture, get_train_loader, get_test_loader, end_function, learning_rate):
    train_loader, validation_loader = get_train_loader(0)
    test_loader = get_test_loader(0)
    net = federated.Local_Model(network_architecture, train_loader, validation_loader, test_loader, N_partitions=0, learning_rate=learning_rate)

    end_condition = False
    epoch_n = 0
    losses = []
    validation_accuracies = []
    accuracies = []
    while False == end_condition:
        (loss_i, local_param_i) = net.train()
        loss = net.get_loss()
        validation_accuracy = net.get_validation_accuracy()
        acc = net.get_accuracy()

        losses.append(loss)
        validation_accuracies.append(validation_accuracy)
        accuracies.append(acc)
        end_condition, optimal_epoch = end_function(epoch_n, validation_accuracies)

        print_results(epoch_n = str(epoch_n), loss=str(loss.item()), validation_accuracy=str(validation_accuracy), acc=str(acc))

        epoch_n = epoch_n + 1
    
    return (optimal_epoch, losses[optimal_epoch], validation_accuracies[optimal_epoch], accuracies[optimal_epoch])

def fed_learning(network_architecture, get_train_loaders, get_test_loaders, N_partitions, end_function, learning_rate):
    train_loaders, validation_loaders = get_train_loaders(N_partitions)
    test_loaders = get_test_loaders(N_partitions)
    local_nets = [federated.Local_Model(network_architecture, train_loaders[node_i], validation_loaders[node_i], test_loaders[node_i], N_partitions, learning_rate) for node_i in range(N_partitions)]
    global_net = federated.Aggregated_Model(network_architecture, N_partitions)

    end_condition = False
    epoch_n = 0
    avg_local_losses = []
    avg_local_validation_accuracies = []
    avg_local_accuracies = []
    while False == end_condition:
        local_params = []
        local_losses = []
        local_validation_accuracies = []
        local_accuracies = []
        for i in range(N_partitions): 
            (loss_i, local_param_i) = local_nets[i].train()
            local_params.append(local_param_i)
            local_losses.append(loss_i)

        global_param = global_net.aggregate_central(local_params)
        
        for i in range(N_partitions): 
            local_nets[i].transfer_param_to_model(global_net.get_params())
            local_validation_accuracies.append(local_nets[i].get_validation_accuracy())
            local_accuracies.append(local_nets[i].get_accuracy())

        avg_local_loss = sum(local_losses) / len(local_losses)
        avg_local_validation_accuracy = sum(local_validation_accuracies) / len(local_validation_accuracies)
        avg_local_accuracy = sum(local_accuracies) / len(local_accuracies)
        
        avg_local_losses.append(avg_local_loss)
        avg_local_validation_accuracies.append(avg_local_validation_accuracy)
        avg_local_accuracies.append(avg_local_accuracy)
        end_condition, optimal_epoch = end_function(epoch_n, avg_local_validation_accuracies)

        print_results(epoch_n = str(epoch_n), loss=str(avg_local_loss.item()), validation_accuracy=str(avg_local_validation_accuracy), acc=str(avg_local_accuracy))
        
        epoch_n = epoch_n + 1

    return (optimal_epoch, avg_local_losses[optimal_epoch], avg_local_validation_accuracies[optimal_epoch], avg_local_accuracies[optimal_epoch])

def selective_aggregation(network_architecture, get_train_loaders, get_test_loaders, N_partitions, end_function, learning_rate):
    with open('config.json') as config_file:
        config = json.load(config_file)
        SIMULARITY_THRESHOLD = float(config['machine_learning']['selective_aggregation']['SIMULARITY_THRESHOLD'])
    
    train_loaders, validation_loaders = get_train_loaders(N_partitions)
    test_loaders = get_test_loaders(N_partitions)
    local_nets = [federated.Local_Model(network_architecture, train_loaders[node_i], validation_loaders[node_i], test_loaders[node_i], N_partitions, learning_rate) for node_i in range(N_partitions)]

    selective_aggregation_vectors = []

    end_condition = False
    epoch_n = 0
    avg_local_losses = []
    avg_local_validation_accuracies = []
    avg_local_accuracies = []
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

        avg_local_losses.append(avg_local_loss)
        avg_local_validation_accuracies.append(avg_local_validation_accuracy)
        avg_local_accuracies.append(avg_local_accuracy)

        end_condition, optimal_epoch = end_function(epoch_n, avg_local_validation_accuracies)
        print_results(epoch_n = str(epoch_n), loss=str(avg_local_loss.item()), validation_accuracy=str(avg_local_validation_accuracy), acc=str(avg_local_accuracy))
        epoch_n = epoch_n + 1
    
    for i in range(len(local_nets)):
        sel_agg_vector = local_nets[i].get_selective_aggregation_vector()
        selective_aggregation_vectors.append(sel_agg_vector)

    print(selective_aggregation_vectors[0].eq(selective_aggregation_vectors[1]).sum().item())
    print(selective_aggregation_vectors[0].eq(selective_aggregation_vectors[2]).sum().item())
    print(selective_aggregation_vectors[0].eq(selective_aggregation_vectors[3]).sum().item())
    print(selective_aggregation_vectors[0].eq(selective_aggregation_vectors[4]).sum().item())
    print(selective_aggregation_vectors[0].eq(selective_aggregation_vectors[5]).sum().item())

    groupings = []
    for i in range(len(selective_aggregation_vectors)):
        if(0 == len(groupings)):
            groupings.append([i])
        else:
            added = False
            for j in range(len(groupings)):
                if (selective_aggregation_vectors[i].eq(selective_aggregation_vectors[groupings[j][0]]).sum().item())/(len(validation_loaders[0].dataset)) > SIMULARITY_THRESHOLD:
                    groupings[j].append(i)
                    added = True
            if False == added:
                groupings.append([i])

    print(groupings)

    global_nets = [federated.Aggregated_Model(network_architecture, N_partitions) for i in range(len(groupings))]

    end_condition = False
    epoch_n = 0
    avg_local_losses = []
    avg_local_validation_accuracies = []
    avg_local_accuracies = []

    while False == end_condition:
        local_losses = []
        local_validation_accuracies = []
        local_accuracies = []
        for i in range(len(groupings)):
            local_params = []
            for j in range(len(groupings[i])):
                (loss_i, local_param_i) = local_nets[groupings[i][j]].train()
                local_params.append(local_param_i)
                local_losses.append(loss_i)

            global_param = global_nets[i].aggregate_central(local_params)
        
            for j in range(len(groupings[i])):
                local_nets[groupings[i][j]].transfer_param_to_model(global_nets[i].get_params())
                local_validation_accuracies.append(local_nets[groupings[i][j]].get_validation_accuracy())
                local_accuracies.append(local_nets[groupings[i][j]].get_accuracy())

        avg_local_loss = sum(local_losses) / len(local_losses)
        avg_local_validation_accuracy = sum(local_validation_accuracies) / len(local_validation_accuracies)
        avg_local_accuracy = sum(local_accuracies) / len(local_accuracies)
        
        avg_local_losses.append(avg_local_loss)
        avg_local_validation_accuracies.append(avg_local_validation_accuracy)
        avg_local_accuracies.append(avg_local_accuracy)
        end_condition, optimal_epoch = end_function(epoch_n, avg_local_validation_accuracies)

        print_results(epoch_n = str(epoch_n), loss=str(avg_local_loss.item()), validation_accuracy=str(avg_local_validation_accuracy), acc=str(avg_local_accuracy))
        
        epoch_n = epoch_n + 1

    return (optimal_epoch, avg_local_losses[optimal_epoch], avg_local_validation_accuracies[optimal_epoch], avg_local_accuracies[optimal_epoch])

def local_learning(network_architecture, get_train_loaders, get_test_loaders, N_partitions, end_function, learning_rate):
    train_loaders, validation_loaders = get_train_loaders(N_partitions)
    test_loaders = get_test_loaders(N_partitions)
    local_nets = [federated.Local_Model(network_architecture, train_loaders[node_i], validation_loaders[node_i], test_loaders[node_i], N_partitions, learning_rate) for node_i in range(N_partitions)]

    end_condition = False
    epoch_n = 0
    avg_local_losses = []
    avg_local_validation_accuracies = []
    avg_local_accuracies = []
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

        avg_local_losses.append(avg_local_loss)
        avg_local_validation_accuracies.append(avg_local_validation_accuracy)
        avg_local_accuracies.append(avg_local_accuracy)
        end_condition, optimal_epoch = end_function(epoch_n, avg_local_validation_accuracies)

        print_results(epoch_n = str(epoch_n), loss=str(avg_local_loss.item()), validation_accuracy=str(avg_local_validation_accuracy), acc=str(avg_local_accuracy))

        epoch_n = epoch_n + 1

    return (optimal_epoch, avg_local_losses[optimal_epoch], avg_local_validation_accuracies[optimal_epoch], avg_local_accuracies[optimal_epoch])

def stop_at_N_epochs_closure(N_epoch):
    def end_N_epochs(n_epoch, measures):
        if(n_epoch < N_epoch - 1):
            return False, N_epoch-1
        else:
            return True, N_epoch-1

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
        EPOCH_SATURATION = int(config['machine_learning']['EPOCH_SATURATION'])
        MAX_EPOCHS = int(config['machine_learning']['MAX_EPOCHS'])

        FC_N_EPOCHS = int(config['machine_learning']['NetFC_1']['FC_N_EPOCHS'])
        FC_BATCH_SIZE = int(config['machine_learning']['NetFC_1']['FC_BATCH_SIZE'])
        FC_LEARNING_RATE = float(config['machine_learning']['NetFC_1']['FC_LEARNING_RATE'])
        C2R3_N_EPOCHS = int(config['machine_learning']['NetCNN_conv2_relu3']['C2R3_N_EPOCHS'])
        C2R3_BATCH_SIZE = int(config['machine_learning']['NetCNN_conv2_relu3']['C2R3_BATCH_SIZE'])
        C2R3_LEARNING_RATE = float(config['machine_learning']['NetCNN_conv2_relu3']['C2R3_LEARNING_RATE'])
        CR3R3_N_EPOCHS = int(config['machine_learning']['NetCNN_convrelu3_relu3']['CR3R3_N_EPOCHS'])
        CR3R3_BATCH_SIZE = int(config['machine_learning']['NetCNN_convrelu3_relu3']['CR3R3_N_EPOCHS'])
        CR3R3_LEARNING_RATE = float(config['machine_learning']['NetCNN_convrelu3_relu3']['CR3R3_LEARNING_RATE'])
    
        SELECTIVE_AGGREGATION_N_PARTITIONS = int(config['machine_learning']['selective_aggregation']['SELECTIVE_AGGREGATION_N_PARTITIONS'])
        SIMILAR_COUNT = int(config['machine_learning']['selective_aggregation']['SIMILAR_COUNT'])
        SIMULARITY_THRESHOLD = float(config['machine_learning']['selective_aggregation']['SIMULARITY_THRESHOLD'])

    federated.set_device("cuda:0")

    stop_at_N_epochs = stop_at_N_epochs_closure(CR3R3_N_EPOCHS)
    stop_at_epoch_saturation = stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION)

    # get_unified_train_loader = data_loader.get_unified_train_loader_closure(FC_BATCH_SIZE)
    # (r_epoch, r_loss, r_val_acc, r_acc) = central_learning(nn_architectures.NetFC_1, get_unified_train_loader, data_loader.get_unified_test_loader, stop_at_N_epochs, FC_LEARNING_RATE)

    # get_random_partitioned_train_loaders = data_loader.get_random_partitioned_train_loaders_closure(FC_BATCH_SIZE)
    # fed_learning(nn_architectures.NetFC_1, get_random_partitioned_train_loaders, data_loader.get_random_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)
    # local_learning(nn_architectures.NetFC_1, get_random_partitioned_train_loaders, data_loader.get_random_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)

    # balance_percentage = 0
    # get_semibalanced_partitioned_train_loader = data_loader.get_semibalanced_partitioned_train_loaders_closure(balance_percentage, C2R3_BATCH_SIZE)   
    # get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(balance_percentage)
    # optimal_epoch, opt_loss_i, opt_val_acc_i, opt_acc_i = fed_learning(nn_architectures.NetCNN_conv2_relu3, get_semibalanced_partitioned_train_loader, 
    #     get_semibalanced_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, C2R3_LEARNING_RATE)

    balance_percentage = 5
    get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(balance_percentage)
    get_selective_aggregation_train_loaders = data_loader.get_selective_aggregation_train_loaders_closure(balance_percentage, FC_BATCH_SIZE, similar_count=3)
    selective_aggregation(nn_architectures.NetFC_1, get_selective_aggregation_train_loaders, get_semibalanced_partitioned_test_loaders, 
        SELECTIVE_AGGREGATION_N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)

if __name__ == "__main__":
    main()


