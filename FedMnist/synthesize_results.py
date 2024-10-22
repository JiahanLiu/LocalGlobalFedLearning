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
SELECTIVE_AGGREGATION_N_PARTITIONS = 0
SIMILAR_COUNT = 0

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
LOCAL_NETFC1_TEST_BALANCED_FILE = ""
LOCAL_NETFC1_TEST_UNBALANCED_FILE = ""
LOCAL_NETC2R3_TEST_BALANCED_FILE = ""
LOCAL_NETC2R3_TEST_UNBALANCED_FILE = ""
LOCAL_NETCR3R3_TEST_BALANCED_FILE = ""
LOCAL_NETCR3R3_TEST_UNBALANCED_FILE = ""
CENTRAL_NETFC1_TEST_BALANCED_FILE = ""
CENTRAL_NETC2R3_TEST_BALANCED_FILE = ""
CENTRAL_NETCR3R3_TEST_BALANCED_FILE = ""
SELECTIVE_FED_NETFC1_TEST_BALANCED_FILE = ""
SELECTIVE_FED_NETC2R3_TEST_BALANCED_FILE = ""
SELECTIVE_FED_NETCR3R3_TEST_BALANCED_FILE = ""
SELECTIVE_FED_NETFC1_TEST_UNBALANCED_FILE = ""
SELECTIVE_FED_NETC2R3_TEST_UNBALANCED_FILE = ""
SELECTIVE_FED_NETCR3R3_TEST_UNBALANCED_FILE = ""

def init():
    global N_PARTITIONS
    global SELECTIVE_AGGREGATION_N_PARTITIONS
    global SIMILAR_COUNT

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
    global LOCAL_NETFC1_TEST_BALANCED_FILE
    global LOCAL_NETFC1_TEST_UNBALANCED_FILE
    global LOCAL_NETC2R3_TEST_BALANCED_FILE 
    global LOCAL_NETC2R3_TEST_UNBALANCED_FILE
    global LOCAL_NETCR3R3_TEST_BALANCED_FILE 
    global LOCAL_NETCR3R3_TEST_UNBALANCED_FILE
    global CENTRAL_NETFC1_TEST_BALANCED_FILE
    global CENTRAL_NETC2R3_TEST_BALANCED_FILE
    global CENTRAL_NETCR3R3_TEST_BALANCED_FILE
    global SELECTIVE_FED_NETFC1_TEST_BALANCED_FILE
    global SELECTIVE_FED_NETC2R3_TEST_BALANCED_FILE
    global SELECTIVE_FED_NETCR3R3_TEST_BALANCED_FILE
    global SELECTIVE_FED_NETFC1_TEST_UNBALANCED_FILE
    global SELECTIVE_FED_NETC2R3_TEST_UNBALANCED_FILE
    global SELECTIVE_FED_NETCR3R3_TEST_UNBALANCED_FILE


    with open('config.json') as config_file:
        config = json.load(config_file)
        N_PARTITIONS = int(config['machine_learning']['N_PARTITIONS'])
        SELECTIVE_AGGREGATION_N_PARTITIONS = int(config['machine_learning']['selective_aggregation']['SELECTIVE_AGGREGATION_N_PARTITIONS'])
        SIMILAR_COUNT = int(config['machine_learning']['selective_aggregation']['SIMILAR_COUNT'])

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
        LOCAL_NETFC1_TEST_BALANCED_FILE = config['results']['LOCAL_NETFC1_TEST_BALANCED_FILE']
        LOCAL_NETFC1_TEST_UNBALANCED_FILE = config['results']['LOCAL_NETFC1_TEST_UNBALANCED_FILE']
        LOCAL_NETC2R3_TEST_BALANCED_FILE = config['results']['LOCAL_NETC2R3_TEST_BALANCED_FILE']
        LOCAL_NETC2R3_TEST_UNBALANCED_FILE = config['results']['LOCAL_NETC2R3_TEST_UNBALANCED_FILE']
        LOCAL_NETCR3R3_TEST_BALANCED_FILE = config['results']['LOCAL_NETCR3R3_TEST_BALANCED_FILE']
        LOCAL_NETCR3R3_TEST_UNBALANCED_FILE = config['results']['LOCAL_NETCR3R3_TEST_UNBALANCED_FILE']
        CENTRAL_NETFC1_TEST_BALANCED_FILE = config['results']['CENTRAL_NETFC1_TEST_BALANCED_FILE']
        CENTRAL_NETC2R3_TEST_BALANCED_FILE = config['results']['CENTRAL_NETC2R3_TEST_BALANCED_FILE']
        CENTRAL_NETCR3R3_TEST_BALANCED_FILE = config['results']['CENTRAL_NETCR3R3_TEST_BALANCED_FILE']
        SELECTIVE_FED_NETFC1_TEST_BALANCED_FILE = config['results']['SELECTIVE_FED_NETFC1_TEST_BALANCED_FILE']
        SELECTIVE_FED_NETC2R3_TEST_BALANCED_FILE = config['results']['SELECTIVE_FED_NETC2R3_TEST_BALANCED_FILE']
        SELECTIVE_FED_NETCR3R3_TEST_BALANCED_FILE = config['results']['SELECTIVE_FED_NETCR3R3_TEST_BALANCED_FILE']
        SELECTIVE_FED_NETFC1_TEST_UNBALANCED_FILE = config['results']['SELECTIVE_FED_NETFC1_TEST_UNBALANCED_FILE']
        SELECTIVE_FED_NETC2R3_TEST_UNBALANCED_FILE = config['results']['SELECTIVE_FED_NETC2R3_TEST_UNBALANCED_FILE']
        SELECTIVE_FED_NETCR3R3_TEST_UNBALANCED_FILE = config['results']['SELECTIVE_FED_NETCR3R3_TEST_UNBALANCED_FILE']

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

def synthesize_balanced_closure(FILEPATH, NetworkArchitecture, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, learning_type):
    def synthesize_balanced(N_averaged, resolution, start_res):
        global N_PARTITIONS

        stop_at_N_epochs = train.stop_at_N_epochs_closure(N_EPOCHS)

        for i in range(start_res, resolution+1):

            balance_percentage = i * 100/resolution

            opt_loss = 0
            opt_val_acc = 0
            opt_acc = 0
            for j in range (N_averaged):
                get_semibalanced_partitioned_train_loader = data_loader.get_semibalanced_partitioned_train_loaders_closure(balance_percentage, BATCH_SIZE)   
                get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(100)
                optimal_epoch, opt_loss_i, opt_val_acc_i, opt_acc_i = learning_type(NetworkArchitecture, get_semibalanced_partitioned_train_loader, 
                    get_semibalanced_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, LEARNING_RATE)
                
                opt_loss = opt_loss + opt_loss_i/N_averaged
                opt_val_acc = opt_val_acc + opt_val_acc_i/N_averaged
                opt_acc = opt_acc + opt_acc_i/N_averaged

            write_results(FILEPATH, balance_percentage, opt_loss.item(), opt_val_acc, opt_acc)
    
    return synthesize_balanced

def synthesize_unbalanced_closure(FILEPATH, NetworkArchitecture, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, learning_type):
    def synthesize_unbalanced(N_averaged, resolution, start_res):
        global N_PARTITIONS

        stop_at_N_epochs = train.stop_at_N_epochs_closure(N_EPOCHS)

        for i in range(start_res, resolution+1):
            balance_percentage = i * 100/resolution

            opt_loss = 0
            opt_val_acc = 0
            opt_acc = 0
            for j in range (N_averaged):
                get_semibalanced_partitioned_train_loader = data_loader.get_semibalanced_partitioned_train_loaders_closure(balance_percentage, BATCH_SIZE)   
                get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(balance_percentage)
                optimal_epoch, opt_loss_i, opt_val_acc_i, opt_acc_i = learning_type(NetworkArchitecture, get_semibalanced_partitioned_train_loader, 
                    get_semibalanced_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, LEARNING_RATE)
                
                opt_loss = opt_loss + opt_loss_i/N_averaged
                opt_val_acc = opt_val_acc + opt_val_acc_i/N_averaged
                opt_acc = opt_acc + opt_acc_i/N_averaged

            write_results(FILEPATH, balance_percentage, opt_loss.item(), opt_val_acc, opt_acc)
        
    return synthesize_unbalanced

def synthesize_selective_aggregation_balanced_closure(FILEPATH, NetworkArchitecture, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, learning_type):
    def synthesize_selective_aggregation_balanced(N_averaged, resolution, start_res):
        global SIMILAR_COUNT
        global SELECTIVE_AGGREGATION_N_PARTITIONS

        stop_at_N_epochs = train.stop_at_N_epochs_closure(N_EPOCHS)

        for i in range(start_res, resolution+1):

            balance_percentage = i * 100/resolution

            opt_loss = 0
            opt_val_acc = 0
            opt_acc = 0
            for j in range (N_averaged):
                get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(100)
                get_selective_aggregation_train_loaders = data_loader.get_selective_aggregation_train_loaders_closure(balance_percentage, 
                    BATCH_SIZE, SIMILAR_COUNT)
                optimal_epoch, opt_loss_i, opt_val_acc_i, opt_acc_i = learning_type(NetworkArchitecture, get_selective_aggregation_train_loaders, 
                    get_semibalanced_partitioned_test_loaders, SELECTIVE_AGGREGATION_N_PARTITIONS, stop_at_N_epochs, LEARNING_RATE)
                
                opt_loss = opt_loss + opt_loss_i/N_averaged
                opt_val_acc = opt_val_acc + opt_val_acc_i/N_averaged
                opt_acc = opt_acc + opt_acc_i/N_averaged

            write_results(FILEPATH, balance_percentage, opt_loss.item(), opt_val_acc, opt_acc)
    
    return synthesize_selective_aggregation_balanced

def synthesize_selective_aggregation_unbalanced_closure(FILEPATH, NetworkArchitecture, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, learning_type):
    def synthesize_selective_aggregation_unbalanced(N_averaged, resolution, start_res):
        global SIMILAR_COUNT
        global SELECTIVE_AGGREGATION_N_PARTITIONS

        stop_at_N_epochs = train.stop_at_N_epochs_closure(N_EPOCHS)

        for i in range(start_res, resolution+1):

            balance_percentage = i * 100/resolution

            opt_loss = 0
            opt_val_acc = 0
            opt_acc = 0
            for j in range (N_averaged):
                get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(balance_percentage)
                get_selective_aggregation_train_loaders = data_loader.get_selective_aggregation_train_loaders_closure(balance_percentage, 
                    BATCH_SIZE, SIMILAR_COUNT)
                optimal_epoch, opt_loss_i, opt_val_acc_i, opt_acc_i = learning_type(NetworkArchitecture, get_selective_aggregation_train_loaders, 
                    get_semibalanced_partitioned_test_loaders, SELECTIVE_AGGREGATION_N_PARTITIONS, stop_at_N_epochs, LEARNING_RATE)
                
                opt_loss = opt_loss + opt_loss_i/N_averaged
                opt_val_acc = opt_val_acc + opt_val_acc_i/N_averaged
                opt_acc = opt_acc + opt_acc_i/N_averaged

            write_results(FILEPATH, balance_percentage, opt_loss.item(), opt_val_acc, opt_acc)
    
    return synthesize_selective_aggregation_unbalanced

def main(): 
    global N_PARTITIONS

    prog_n = -1
    gpu_n = -1
    start_res = -1
    locality = -1
    options, remainder = getopt.getopt(sys.argv[1:], 'p:g:s:l:')
    for opt, arg in options:
        if opt in ('-p'):
            prog_n = int(arg)
        if opt in ('-g'):
            gpu_n = int(arg)
        if opt in ('-s'):
            start_res = int(arg)
        if opt in ('-l'):
            locality = int(arg)

    init()

    N_averaged = 1
    resolution = 50

    federated.set_device("cuda:" + str(gpu_n%torch.cuda.device_count()))
    if(-1 == prog_n):
        print("Add Arg for GPU number or Re-write code to do on CPU")
    elif (0 == prog_n):
        if (1 == locality):
            synthesize_balanced = synthesize_balanced_closure(FED_NETFC1_TEST_BALANCED_FILE, nn_architectures.NetFC_1, 
                FC_N_EPOCHS, FC_BATCH_SIZE, FC_LEARNING_RATE, train.fed_learning)
        if (2 == locality):
            synthesize_balanced = synthesize_balanced_closure(LOCAL_NETFC1_TEST_BALANCED_FILE, nn_architectures.NetFC_1, 
                FC_N_EPOCHS, FC_BATCH_SIZE, FC_LEARNING_RATE, train.local_learning)
        if (3 == locality):
            synthesize_balanced = synthesize_selective_aggregation_balanced_closure(SELECTIVE_FED_NETFC1_TEST_BALANCED_FILE, 
                nn_architectures.NetFC_1, FC_N_EPOCHS, FC_BATCH_SIZE, FC_LEARNING_RATE, train.selective_aggregation)
            
        synthesize_balanced(N_averaged, resolution, start_res)

    elif (1 == prog_n):
        if (1 == locality):
            synthesize_unbalanced = synthesize_unbalanced_closure(FED_NETFC1_TEST_UNBALANCED_FILE, nn_architectures.NetFC_1, 
                FC_N_EPOCHS, FC_BATCH_SIZE, FC_LEARNING_RATE, train.fed_learning)
        if (2 == locality):
            synthesize_unbalanced = synthesize_unbalanced_closure(LOCAL_NETFC1_TEST_UNBALANCED_FILE, nn_architectures.NetFC_1, 
                FC_N_EPOCHS, FC_BATCH_SIZE, FC_LEARNING_RATE, train.local_learning)
        if (3 == locality):
            synthesize_unbalanced = synthesize_selective_aggregation_unbalanced_closure(SELECTIVE_FED_NETFC1_TEST_UNBALANCED_FILE, 
                nn_architectures.NetFC_1, FC_N_EPOCHS, FC_BATCH_SIZE, FC_LEARNING_RATE, train.selective_aggregation)
            
        synthesize_unbalanced(N_averaged, resolution, start_res)

    elif (2 == prog_n):
        if (1 == locality):
            synthesize_balanced = synthesize_balanced_closure(FED_NETC2R3_TEST_BALANCED_FILE, nn_architectures.NetCNN_conv2_relu3, 
                C2R3_N_EPOCHS, C2R3_BATCH_SIZE, C2R3_LEARNING_RATE, train.fed_learning)
        if (2 == locality):
            synthesize_balanced = synthesize_balanced_closure(LOCAL_NETC2R3_TEST_BALANCED_FILE, nn_architectures.NetCNN_conv2_relu3, 
                C2R3_N_EPOCHS, C2R3_BATCH_SIZE, C2R3_LEARNING_RATE, train.local_learning)
        if (3 == locality):
            synthesize_balanced = synthesize_selective_aggregation_balanced_closure(SELECTIVE_FED_NETC2R3_TEST_BALANCED_FILE, 
                nn_architectures.NetCNN_conv2_relu3, C2R3_N_EPOCHS, C2R3_BATCH_SIZE, C2R3_LEARNING_RATE, train.selective_aggregation)
        
        synthesize_balanced(N_averaged, resolution, start_res)

    elif (3 == prog_n):
        if (1 == locality):
            synthesize_unbalanced = synthesize_unbalanced_closure(FED_NETC2R3_TEST_UNBALANCED_FILE, nn_architectures.NetCNN_conv2_relu3, 
                C2R3_N_EPOCHS, C2R3_BATCH_SIZE, C2R3_LEARNING_RATE, train.fed_learning)
        if (2 == locality):
            synthesize_unbalanced = synthesize_unbalanced_closure(LOCAL_NETC2R3_TEST_UNBALANCED_FILE, nn_architectures.NetCNN_conv2_relu3, 
                C2R3_N_EPOCHS, C2R3_BATCH_SIZE, C2R3_LEARNING_RATE, train.local_learning)
        if (3 == locality):
            synthesize_unbalanced = synthesize_selective_aggregation_unbalanced_closure(SELECTIVE_FED_NETC2R3_TEST_UNBALANCED_FILE, 
                nn_architectures.NetCNN_conv2_relu3, C2R3_N_EPOCHS, C2R3_BATCH_SIZE, C2R3_LEARNING_RATE, train.selective_aggregation)
        
        synthesize_unbalanced(N_averaged, resolution, start_res)

    elif (4 == prog_n):
        if (1 == locality):
            synthesize_balanced = synthesize_balanced_closure(FED_NETCR3R3_TEST_BALANCED_FILE, nn_architectures.NetCNN_convrelu3_relu3, 
                CR3R3_N_EPOCHS, CR3R3_BATCH_SIZE, CR3R3_LEARNING_RATE, train.fed_learning)
        if (2 == locality):
            synthesize_balanced = synthesize_balanced_closure(LOCAL_NETCR3R3_TEST_BALANCED_FILE, nn_architectures.NetCNN_convrelu3_relu3, 
                CR3R3_N_EPOCHS, CR3R3_BATCH_SIZE, CR3R3_LEARNING_RATE, train.local_learning)
        if (3 == locality):
            synthesize_balanced = synthesize_selective_aggregation_balanced_closure(SELECTIVE_FED_NETCR3R3_TEST_BALANCED_FILE, 
            nn_architectures.NetCNN_convrelu3_relu3, CR3R3_N_EPOCHS, CR3R3_BATCH_SIZE, CR3R3_LEARNING_RATE, train.selective_aggregation)
            
        synthesize_balanced(N_averaged, resolution, start_res)

    elif (5 == prog_n):
        if (1 == locality):
            synthesize_unbalanced = synthesize_unbalanced_closure(FED_NETCR3R3_TEST_UNBALANCED_FILE, nn_architectures.NetCNN_convrelu3_relu3, 
                CR3R3_N_EPOCHS, CR3R3_BATCH_SIZE, CR3R3_LEARNING_RATE, train.fed_learning)
        if (2 == locality):
            synthesize_unbalanced = synthesize_unbalanced_closure(LOCAL_NETCR3R3_TEST_UNBALANCED_FILE, nn_architectures.NetCNN_convrelu3_relu3, 
                CR3R3_N_EPOCHS, CR3R3_BATCH_SIZE, CR3R3_LEARNING_RATE, train.local_learning)
        if (3 == locality):
            synthesize_unbalanced = synthesize_selective_aggregation_unbalanced_closure(SELECTIVE_FED_NETCR3R3_TEST_UNBALANCED_FILE, 
            nn_architectures.NetCNN_convrelu3_relu3, CR3R3_N_EPOCHS, CR3R3_BATCH_SIZE, CR3R3_LEARNING_RATE, train.selective_aggregation)

        synthesize_unbalanced(N_averaged, resolution, start_res)

    else:
        print("Invalid Arg: " + str(gpu_n))

if __name__ == "__main__":
    main()
