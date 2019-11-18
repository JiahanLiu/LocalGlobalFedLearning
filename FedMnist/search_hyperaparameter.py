from model import nn_architectures, data_loader
import train

import csv
import json
import math
import os.path
import random

N_PARTITIONS = 0
EPOCH_SATURATION = 0
MAX_EPOCHS = 0
RANDOM_SEED = 0
CENTRAL_NETFC1_BALANCED100_FILE = ""

def init():
    global N_PARTITIONS
    global EPOCH_SATURATION
    global MAX_EPOCHS
    global RANDOM_SEED
    global CENTRAL_NETFC1_BALANCED100_FILE
    with open('config.json') as config_file:
        config = json.load(config_file)
        N_PARTITIONS = int(config['machine_learning']['N_PARTITIONS'])
        EPOCH_SATURATION = int(config['machine_learning']['EPOCH_SATURATION'])
        MAX_EPOCHS = int(config['machine_learning']['MAX_EPOCHS'])
        RANDOM_SEED = int(config['machine_learning']['RANDOM_SEED'])

        CENTRAL_NETFC1_BALANCED100_FILE = config['hyperparameter_search']['CENTRAL_NETFC1_BALANCED100_FILE']

def print_results(optimal_epoch, opt_loss, opt_validation_accuracy, opt_acc):
    print("Opt Epoch: " + optimal_epoch + " | Opt Loss: " + opt_loss + " | Opt Val Acc" + opt_validation_accuracy + " | Opt Acc: " + opt_acc)

def search_fed_model(n_iterations):
    for i in range(n_iterations):
        optimal_epoch, random_batch_size, random_learning_rate, opt_loss, opt_val_acc, opt_acc = search_fed_model_single()
        write_results(CENTRAL_NETFC1_BALANCED100_FILE, str(optimal_epoch), str(random_batch_size), str(random_learning_rate), str(opt_loss), str(opt_val_acc), str(opt_acc))

def search_fed_model_single():
    random.seed(RANDOM_SEED)
    stop_at_epoch_saturation = train.stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION)

    random_learning_rate = (random.random() * 100) * math.pow(10, -4)
    random_batch_size = random.randrange(10, 1000)

    get_random_partitioned_train_loaders = data_loader.get_random_partitioned_train_loaders_closure(batch_size=random_batch_size)
    optimal_epoch, opt_loss, opt_val_acc, opt_acc = train.fed_learning(nn_architectures.NetFC_1, get_random_partitioned_train_loaders, 
        data_loader.get_random_partitioned_test_loaders, N_PARTITIONS, stop_at_epoch_saturation, random_learning_rate)

    print_results(optimal_epoch=str(optimal_epoch), opt_loss=str(opt_loss.item()), opt_validation_accuracy=str(opt_val_acc), opt_acc=str(opt_acc))

    return optimal_epoch, random_batch_size, random_learning_rate, opt_loss.item, opt_val_acc, opt_acc

def write_results(file_path, optimal_epoch, batch_size, learning_rate, opt_loss, opt_val_acc, opt_acc):
    if False == os.path.isfile(file_path):
        title_row=['optimal_epoch','batch_size','learning_rate', 'opt_loss', 'opt_val_acc', 'opt_acc']
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(title_row)

    result_row=[optimal_epoch, batch_size, learning_rate, opt_loss, opt_val_acc, opt_acc]
    with open(file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

def main(): 
    init()

    search_fed_model(1)
    # # for i in range(0, 30):
    # random_learning_rate = (random.random() * 100) * math.pow(10, -4)
    # random_batch_size = random.randrange(10, 1000)
    # get_unified_train_loader = data_loader.get_unified_train_loader_closure(batch_size=256)
    # optimal_epoch, optimal_validation_accuracy = train.fed_learning(nn_architectures.NetFC_1, get_unified_train_loader, data_loader.get_unified_test_loader, stop_at_epoch_saturation, 0.001)

    # print(optimal_epoch)
    # print(optimal_validation_accuracy)
 
    # title_row=['validation_accuracy','N_EPOCH','BATCH_SIZE', 'LEARNING_RATE']
    # result_row=[optimal_validation_accuracy, optimal_epoch, random_batch_size, random_learning_rate]
    # with open('hyperparameter_search/central_NetFC_1.csv', 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(title_row)
    #     writer.writerow(result_row)

if __name__ == "__main__":
    main()