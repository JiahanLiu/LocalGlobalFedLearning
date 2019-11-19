import federated
from model import nn_architectures, data_loader
import train

import torch

import csv
import fcntl
import getopt
import json
import math
from multiprocessing import Pool
import os.path
import random
import sys

N_PARTITIONS = 0
EPOCH_SATURATION = 0
MAX_EPOCHS = 0
FED_NETFC1_BALANCED100_FILE = ""

def init():
    global N_PARTITIONS
    global EPOCH_SATURATION
    global MAX_EPOCHS
    global FED_NETFC1_BALANCED100_FILE
    with open('config.json') as config_file:
        config = json.load(config_file)
        N_PARTITIONS = int(config['machine_learning']['N_PARTITIONS'])
        EPOCH_SATURATION = int(config['machine_learning']['EPOCH_SATURATION'])
        MAX_EPOCHS = int(config['machine_learning']['MAX_EPOCHS'])

        FED_NETFC1_BALANCED100_FILE = config['hyperparameter_search']['FED_NETFC1_BALANCED100_FILE']

def print_results(optimal_epoch, batch_size, learning_rate, opt_loss, opt_validation_accuracy, opt_acc):
    print("Opt Epoch: " + optimal_epoch + " | Batch Size: " + batch_size + " | Learning Rate: " + learning_rate + " | Opt Loss: " + opt_loss 
        + " | Opt Val Acc: " + opt_validation_accuracy + " | Opt Acc: " + opt_acc)

def search_fed_model(n_iterations, gpu_n, N_averaged):
    federated.set_device("cuda:" + str(gpu_n))
    optimal_epoch=opt_loss=opt_val_acc=opt_acc=0
    for i in range(n_iterations):
        for j in range(N_averaged):
            optimal_epoch_i, random_batch_size, random_learning_rate, opt_loss_i, opt_val_acc_i, opt_acc_i = search_fed_model_single()
            optimal_epoch = optimal_epoch_i + optimal_epoch_i/N_averaged
            opt_loss = opt_loss_i + opt_loss/N_averaged
            opt_val_acc = opt_val_acc + optimal_epoch_i/N_averaged
            opt_acc = opt_acc + optimal_epoch_i/N_averaged
    write_results(FED_NETFC1_BALANCED100_FILE, str(optimal_epoch), str(random_batch_size), str(random_learning_rate), str(opt_loss), str(opt_val_acc), str(opt_acc))

def search_fed_model_single():
    stop_at_epoch_saturation = train.stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION)

    batch_size_options = [64, 128, 256]

    random_learning_rate = (random.random() * 1000) * math.pow(10, -5)
    random_batch_size = batch_size_options[random.randrange(0, len(batch_size_options))]

    get_random_partitioned_train_loaders = data_loader.get_random_partitioned_train_loaders_closure(batch_size=random_batch_size)
    optimal_epoch, opt_loss, opt_val_acc, opt_acc = train.fed_learning(nn_architectures.NetFC_1, get_random_partitioned_train_loaders, 
        data_loader.get_random_partitioned_test_loaders, N_PARTITIONS, stop_at_epoch_saturation, random_learning_rate)

    print_results(optimal_epoch=str(optimal_epoch), batch_size=str(random_batch_size), learning_rate=str(random_learning_rate),  
        opt_loss=str(opt_loss.item()), opt_validation_accuracy=str(opt_val_acc), opt_acc=str(opt_acc))

    return optimal_epoch, random_batch_size, random_learning_rate, opt_loss.item(), opt_val_acc, opt_acc

def write_results(file_path, optimal_epoch, batch_size, learning_rate, opt_loss, opt_val_acc, opt_acc):
    if False == os.path.isfile(file_path):
        title_row=['optimal_epoch','batch_size','learning_rate', 'opt_loss', 'opt_val_acc', 'opt_acc']
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(title_row)

    result_row=[optimal_epoch, batch_size, learning_rate, opt_loss, opt_val_acc, opt_acc]
    with open(file_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.writer(f)
        writer.writerow(result_row)
        fcntl.flock(f, fcntl.LOCK_UN) 

def pull_top_results(file_path):
    opt_20_list = []
    with open(file_path, 'r') as f:
        readCSV = csv.reader(f)
        for index, row in enumerate(readCSV):
            if 0 == index:
                doNothing = 0
            elif index < 20+1:
                opt_20_list.append([float(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
            else:
                opt_20_list.sort(key = lambda x: x[4], reverse=True)
                if (opt_20_list[19][4] < float(row[4])) and (float(row[1]) == 128):
                    opt_20_list[19] = [float(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
    
    for row in opt_20_list:
        print_results(str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5]))
            

def main(): 
    options, remainder = getopt.getopt(sys.argv[1:], 'g:')
    for opt, arg in options:
        if opt in ('-g'):
            gpu_n = arg

    init()
    # search_fed_model(100, gpu_n, N_averaged=5)
    pull_top_results(FED_NETFC1_BALANCED100_FILE)

if __name__ == "__main__":
    main()
