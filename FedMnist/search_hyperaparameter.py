from model import nn_architectures, data_loader
import train

import csv
import json
import math
import random

def main(): 
    with open('config.json') as config_file:
        config = json.load(config_file)
        EPOCH_SATURATION = int(config['machine_learning']['EPOCH_SATURATION'])
        MAX_EPOCHS = int(config['machine_learning']['MAX_EPOCHS'])

    stop_at_epoch_saturation = train.stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION)

    random.seed(90)
    # for i in range(0, 30):
    random_learning_rate = (random.random() * 100) * math.pow(10, -4)
    random_batch_size = random.randrange(10, 1000)
    get_unified_train_loader = data_loader.get_unified_train_loader_closure(batch_size=256)
    optimal_epoch, optimal_validation_accuracy = train.central_learning(nn_architectures.NetFC_1, get_unified_train_loader, data_loader.get_unified_test_loader, stop_at_epoch_saturation, 0.001)

    print(optimal_epoch)
    print(optimal_validation_accuracy)
 
    title_row=['validation_accuracy','N_EPOCH','BATCH_SIZE', 'LEARNING_RATE']
    result_row=[optimal_validation_accuracy, optimal_epoch, random_batch_size, random_learning_rate]
    with open('hyperparameter_search/central_NetFC_1.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(title_row)
        writer.writerow(result_row)

    

if __name__ == "__main__":
    main()