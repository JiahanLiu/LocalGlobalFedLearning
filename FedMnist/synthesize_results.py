import federated
from model import nn_architectures, data_loader
import train

import torch

import csv
import getopt
import json
import matplotlib.pyplot as plt
import os
import sys

N_PARTITIONS = 0

FC_N_EPOCHS = 0
FC_BATCH_SIZE = 0
FC_LEARNING_RATE = 0
C2R3_N_EPOCHS = 0
C2R3_BATCH_SIZE = 0
C2R3_LEARNING_RATE = 0
CR3R3_N_EPOCHS = 0
CR3R3_BATCH_SIZE = 0
CR3R3_LEARNING_RATE = 0

FED_NETFC1_TEST_BALANCED_FILE = ""
FED_NETFC1_TEST_UNBALANCED_FILE = ""
FED_NETC2R3_TEST_BALANCED_FILE = ""
FED_NETC2R3_TEST_UNBALANCED_FILE = ""
FED_NETCR3R3_TEST_BALANCED_FILE = ""
FED_NETCR3R3_TEST_UNBALANCED_FILE = ""

def init():
    global N_PARTITIONS

    global FC_N_EPOCHS
    global FC_BATCH_SIZE
    global FC_LEARNING_RATE
    global C2R3_N_EPOCHS
    global C2R3_BATCH_SIZE
    global C2R3_LEARNING_RATE
    global CR3R3_N_EPOCHS
    global CR3R3_BATCH_SIZE
    global CR3R3_LEARNING_RATE

    global FED_NETFC1_TEST_BALANCED_FILE
    global FED_NETFC1_TEST_UNBALANCED_FILE
    global FED_NETC2R3_TEST_BALANCED_FILE
    global FED_NETC2R3_TEST_UNBALANCED_FILE
    global FED_NETCR3R3_TEST_BALANCED_FILE
    global FED_NETCR3R3_TEST_UNBALANCED_FILE

    with open('config.json') as config_file:
        config = json.load(config_file)
        N_PARTITIONS = int(config['machine_learning']['N_PARTITIONS'])

        FC_N_EPOCHS = int(config['machine_learning']['NetFC_1']['FC_N_EPOCHS'])
        FC_BATCH_SIZE = int(config['machine_learning']['NetFC_1']['FC_BATCH_SIZE'])
        FC_LEARNING_RATE = float(config['machine_learning']['NetFC_1']['FC_LEARNING_RATE'])
        C2R3_N_EPOCHS = int(config['machine_learning']['NetCNN_conv2_relu3']['C2R3_N_EPOCHS'])
        C2R3_BATCH_SIZE = int(config['machine_learning']['NetCNN_conv2_relu3']['C2R3_BATCH_SIZE'])
        C2R3_LEARNING_RATE = float(config['machine_learning']['NetCNN_conv2_relu3']['C2R3_LEARNING_RATE'])
        CR3R3_N_EPOCHS = int(config['machine_learning']['NetCNN_convrelu3_relu3']['CR3R3_N_EPOCHS'])
        CR3R3_BATCH_SIZE = int(config['machine_learning']['NetCNN_convrelu3_relu3']['CR3R3_N_EPOCHS'])
        CR3R3_LEARNING_RATE = float(config['machine_learning']['NetCNN_convrelu3_relu3']['CR3R3_LEARNING_RATE'])

        FED_NETFC1_TEST_BALANCED_FILE = config['results']['FED_NETFC1_TEST_BALANCED_FILE']
        FED_NETFC1_TEST_UNBALANCED_FILE = config['results']['FED_NETFC1_TEST_UNBALANCED_FILE']
        FED_NETC2R3_TEST_BALANCED_FILE = config['results']['FED_NETC2R3_TEST_BALANCED_FILE']
        FED_NETC2R3_TEST_UNBALANCED_FILE = config['results']['FED_NETC2R3_TEST_UNBALANCED_FILE']
        FED_NETCR3R3_TEST_BALANCED_FILE = config['results']['FED_NETCR3R3_TEST_BALANCED_FILE']
        FED_NETCR3R3_TEST_UNBALANCED_FILE = config['results']['FED_NETCR3R3_TEST_UNBALANCED_FILE']

def write_results(file_path, balance_percentage, loss, validation_accuracy, accuracy):
    if False == os.path.isfile(file_path):
        title_row=['Balanced Percentage','Loss', 'Validation Accuracy', 'Accuracy']
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(title_row)

    result_row=[balance_percentage, loss, validation_accuracy, accuracy]
    with open(file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

def synthesize_balanced_closure(FILEPATH, NetworkArchitecture, N_EPOCHS, BATCH_SIZE, LEARNING_RATE):
    def synthesize_balanced(N_averaged, resolution):
        global N_PARTITIONS

        stop_at_N_epochs = train.stop_at_N_epochs_closure(N_EPOCHS)

        for i in range(resolution+1):
            balance_percentage = i * 100/resolution

            opt_loss = 0
            opt_val_acc = 0
            opt_acc = 0
            for i in range (N_averaged):
                get_semibalanced_partitioned_train_loader = data_loader.get_semibalanced_partitioned_train_loaders_closure(balance_percentage, BATCH_SIZE)   
                get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(100)
                optimal_epoch, opt_loss_i, opt_val_acc_i, opt_acc_i = train.fed_learning(NetworkArchitecture, get_semibalanced_partitioned_train_loader, 
                    get_semibalanced_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, LEARNING_RATE)
                
                opt_loss = opt_loss + opt_loss_i/N_averaged
                opt_val_acc = opt_val_acc + opt_val_acc_i/N_averaged
                opt_acc = opt_acc + opt_acc_i/N_averaged

            write_results(FILEPATH, balance_percentage, opt_loss.item(), opt_val_acc, opt_acc)
    
    return synthesize_balanced

def synthesize_unbalanced_closure(FILEPATH, NetworkArchitecture, N_EPOCHS, BATCH_SIZE, LEARNING_RATE):
    def synthesize_unbalanced(N_averaged, resolution):
        global N_PARTITIONS

        stop_at_N_epochs = train.stop_at_N_epochs_closure(N_EPOCHS)

        for i in range(resolution+1):
            balance_percentage = i * 100/resolution

            opt_loss = 0
            opt_val_acc = 0
            opt_acc = 0
            for i in range (N_averaged):
                get_semibalanced_partitioned_train_loader = data_loader.get_semibalanced_partitioned_train_loaders_closure(balance_percentage, BATCH_SIZE)   
                get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(balance_percentage)
                optimal_epoch, opt_loss_i, opt_val_acc_i, opt_acc_i = train.fed_learning(NetworkArchitecture, get_semibalanced_partitioned_train_loader, 
                    get_semibalanced_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, LEARNING_RATE)
                
                opt_loss = opt_loss + opt_loss_i/N_averaged
                opt_val_acc = opt_val_acc + opt_val_acc_i/N_averaged
                opt_acc = opt_acc + opt_acc_i/N_averaged

            write_results(FILEPATH, balance_percentage, opt_loss.item(), opt_val_acc, opt_acc)
        
    return synthesize_unbalanced

def main(): 
    gpu_n = -1
    options, remainder = getopt.getopt(sys.argv[1:], 'g:')
    for opt, arg in options:
        if opt in ('-g'):
            gpu_n = int(arg)

    init()

    N_averaged = 1

    if(-1 == gpu_n):
        print("Add Arg for GPU number or Re-write code to do on CPU")
    elif (0 == gpu_n):
        federated.set_device("cuda:" + str(gpu_n%torch.cuda.device_count()))
        synthesize_balanced = synthesize_balanced_closure(FED_NETFC1_TEST_BALANCED_FILE, nn_architectures.NetFC_1, 
            FC_N_EPOCHS, FC_BATCH_SIZE, FC_LEARNING_RATE)
        synthesize_balanced(N_averaged, resolution=50)

    elif (1 == gpu_n):
        federated.set_device("cuda:" + str(gpu_n%torch.cuda.device_count()))
        synthesize_unbalanced = synthesize_unbalanced_closure(FED_NETFC1_TEST_UNBALANCED_FILE, nn_architectures.NetFC_1, 
            FC_N_EPOCHS, FC_BATCH_SIZE, FC_LEARNING_RATE)
        synthesize_unbalanced(N_averaged, resolution=50)

    elif (2 == gpu_n):
        federated.set_device("cuda:" + str(gpu_n%torch.cuda.device_count()))
        synthesize_balanced = synthesize_balanced_closure(FED_NETC2R3_TEST_BALANCED_FILE, nn_architectures.NetCNN_conv2_relu3, 
            C2R3_N_EPOCHS, C2R3_BATCH_SIZE, C2R3_LEARNING_RATE)
        synthesize_balanced(N_averaged, resolution=50)

    elif (3 == gpu_n):
        federated.set_device("cuda:" + str(gpu_n%torch.cuda.device_count()))
        synthesize_unbalanced = synthesize_unbalanced_closure(FED_NETC2R3_TEST_UNBALANCED_FILE, nn_architectures.NetCNN_conv2_relu3, 
            C2R3_N_EPOCHS, C2R3_BATCH_SIZE, C2R3_LEARNING_RATE)
        synthesize_unbalanced(N_averaged, resolution=50)

    elif (4 == gpu_n):
        federated.set_device("cuda:" + str(gpu_n%torch.cuda.device_count()))
        synthesize_unbalanced = synthesize_unbalanced_closure(FED_NETCR3R3_TEST_UNBALANCED_FILE, nn_architectures.NetCNN_convrelu3_relu3, 
            CR3R3_N_EPOCHS, CR3R3_BATCH_SIZE, CR3R3_LEARNING_RATE)
        synthesize_unbalanced(N_averaged, resolution=50)

    elif (5 == gpu_n):
        federated.set_device("cuda:" + str(gpu_n%torch.cuda.device_count()))
        synthesize_balanced = synthesize_balanced_closure(FED_NETCR3R3_TEST_BALANCED_FILE, nn_architectures.NetCNN_convrelu3_relu3, 
            CR3R3_N_EPOCHS, CR3R3_BATCH_SIZE, CR3R3_LEARNING_RATE)
        synthesize_balanced(N_averaged, resolution=50)
    else:
        print("Invalid Arg: " + str(gpu_n))

if __name__ == "__main__":
    main()
