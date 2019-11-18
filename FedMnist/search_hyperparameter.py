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
CENTRAL_NETFC1_BALANCED100_FILE = ""

def init():
    global N_PARTITIONS
    global EPOCH_SATURATION
    global MAX_EPOCHS
    global CENTRAL_NETFC1_BALANCED100_FILE
    with open('config.json') as config_file:
        config = json.load(config_file)
        N_PARTITIONS = int(config['machine_learning']['N_PARTITIONS'])
        EPOCH_SATURATION = int(config['machine_learning']['EPOCH_SATURATION'])
        MAX_EPOCHS = int(config['machine_learning']['MAX_EPOCHS'])

        CENTRAL_NETFC1_BALANCED100_FILE = config['hyperparameter_search']['CENTRAL_NETFC1_BALANCED100_FILE']

def print_results(optimal_epoch, batch_size, learning_rate, opt_loss, opt_validation_accuracy, opt_acc):
    print("Opt Epoch: " + optimal_epoch + " | Batch Size: " + batch_size + " | Learning Rate: " + learning_rate + " | Opt Loss: " + opt_loss 
        + " | Opt Val Acc: " + opt_validation_accuracy + " | Opt Acc: " + opt_acc)

def search_fed_model(n_iterations, gpu_n):
    federated.set_device("cuda:" + str(gpu_n))
    for i in range(n_iterations):
        optimal_epoch, random_batch_size, random_learning_rate, opt_loss, opt_val_acc, opt_acc = search_fed_model_single()
        write_results(CENTRAL_NETFC1_BALANCED100_FILE, str(optimal_epoch), str(random_batch_size), str(random_learning_rate), str(opt_loss), str(opt_val_acc), str(opt_acc))


def search_fed_model_single():
    stop_at_epoch_saturation = train.stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION)

    random_learning_rate = (random.random() * 10000) * math.pow(10, -5)
    random_batch_size = random.randrange(5, 5000)

    get_random_partitioned_train_loaders = data_loader.get_random_partitioned_train_loaders_closure(batch_size=random_batch_size)
    optimal_epoch, opt_loss, opt_val_acc, opt_acc = train.fed_learning(nn_architectures.NetFC_1, get_random_partitioned_train_loaders, 
        data_loader.get_random_partitioned_test_loaders, N_PARTITIONS, stop_at_epoch_saturation, random_learning_rate)

    print_results(optimal_epoch=str(optimal_epoch), batch_size=str(random_batch_size), learning_rate=str(random_learning_rate),  
        opt_loss=str(opt_loss.item()), opt_validation_accuracy=str(opt_val_acc), opt_acc=str(opt_acc))

    return optimal_epoch, random_batch_size, random_learning_rate, opt_loss.item(), opt_val_acc, opt_acc

def write_results(file_path, optimal_epoch, batch_size, learning_rate, opt_loss, opt_val_acc, opt_acc):
    if False == os.path.isfile(file_path):
        title_row=['optimal_epoch','batch_size','learning_rate', 'opt_loss', 'opt_val_acc', 'opt_acc']
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(title_row)

    result_row=[optimal_epoch, batch_size, learning_rate, opt_loss, opt_val_acc, opt_acc]
    with open(file_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.writer(f)
        writer.writerow(result_row)
        fcntl.flock(f, fcntl.LOCK_UN) 

def main(): 
    options, remainder = getopt.getopt(sys.argv[1:], 'g:')
    for opt, arg in options:
        if opt in ('-g'):
            gpu_n = arg

    init()
    search_fed_model(100, gpu_n)

if __name__ == "__main__":
    main()
