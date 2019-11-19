import federated
from model import nn_architectures, data_loader
import train

import csv
import json
import matplotlib.pyplot as plt
import os

N_PARTITIONS = 0
N_EPOCHS = 0
BATCH_SIZE = 0
FC_LEARNING_RATE = 0
FED_NETFC1_TEST_BALANCED_FILE = ""
FED_NETFC1_TEST_UNBALANCED_FILE = ""

def init():
    global N_PARTITIONS
    global N_EPOCHS
    global BATCH_SIZE
    global FC_LEARNING_RATE
    global FED_NETFC1_TEST_BALANCED_FILE
    global FED_NETFC1_TEST_UNBALANCED_FILE
    with open('config.json') as config_file:
        config = json.load(config_file)
        N_PARTITIONS = int(config['machine_learning']['N_PARTITIONS'])
        N_EPOCHS = int(config['machine_learning']['N_EPOCHS'])
        BATCH_SIZE = int(config['machine_learning']['BATCH_SIZE'])
        FC_LEARNING_RATE = float(config['machine_learning']['FC_LEARNING_RATE'])

        FED_NETFC1_TEST_BALANCED_FILE = config['results']['FED_NETFC1_TEST_BALANCED_FILE']
        FED_NETFC1_TEST_UNBALANCED_FILE = config['results']['FED_NETFC1_TEST_UNBALANCED_FILE']

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

def synthesize_NETFC1_test_balanced(N_averaged, resolution):
    global N_PARTITIONS
    global N_EPOCHS
    global BATCH_SIZE
    global FC_LEARNING_RATE

    stop_at_N_epochs = train.stop_at_N_epochs_closure(N_EPOCHS)

    for i in range(resolution+1):
        balance_percentage = i * 100/resolution

        opt_loss = 0
        opt_val_acc = 0
        opt_acc = 0
        for i in range (N_averaged):
            get_semibalanced_partitioned_train_loader = data_loader.get_semibalanced_partitioned_train_loaders_closure(balance_percentage, BATCH_SIZE)   
            get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(100)
            optimal_epoch, opt_loss_i, opt_val_acc_i, opt_acc_i = train.fed_learning(nn_architectures.NetFC_1, get_semibalanced_partitioned_train_loader, 
                get_semibalanced_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)
            
            opt_loss = opt_loss + opt_loss_i/N_averaged
            opt_val_acc = opt_val_acc + opt_val_acc_i/N_averaged
            opt_acc = opt_acc + opt_acc_i/N_averaged

        write_results(FED_NETFC1_TEST_BALANCED_FILE, balance_percentage, opt_loss.item(), opt_val_acc, opt_acc)

def synthesize_NETFC1_test_unbalanced(N_averaged, resolution):
    global N_PARTITIONS
    global N_EPOCHS
    global BATCH_SIZE
    global FC_LEARNING_RATE

    stop_at_N_epochs = train.stop_at_N_epochs_closure(N_EPOCHS)

    for i in range(resolution+1):
        balance_percentage = i * 100/resolution

        opt_loss = 0
        opt_val_acc = 0
        opt_acc = 0
        for i in range (N_averaged):
            get_semibalanced_partitioned_train_loader = data_loader.get_semibalanced_partitioned_train_loaders_closure(balance_percentage, BATCH_SIZE)   
            get_semibalanced_partitioned_test_loaders = data_loader.get_semibalanced_partitioned_test_loaders_closure(balance_percentage)
            optimal_epoch, opt_loss_i, opt_val_acc_i, opt_acc_i = train.fed_learning(nn_architectures.NetFC_1, get_semibalanced_partitioned_train_loader, 
                get_semibalanced_partitioned_test_loaders, N_PARTITIONS, stop_at_N_epochs, FC_LEARNING_RATE)
            
            opt_loss = opt_loss + opt_loss_i/N_averaged
            opt_val_acc = opt_val_acc + opt_val_acc_i/N_averaged
            opt_acc = opt_acc + opt_acc_i/N_averaged

        write_results(FED_NETFC1_TEST_UNBALANCED_FILE, balance_percentage, opt_loss.item(), opt_val_acc, opt_acc)

def main(): 
    init()

    synthesize_NETFC1_test_balanced(N_averaged=3, resolution=20)
    synthesize_NETFC1_test_unbalanced(N_averaged=3, resolution=20)

if __name__ == "__main__":
    main()
