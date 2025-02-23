import evaluate
from model import nn_architectures

import torch
import torch.nn as nn
import torch.optim as optim

import json
import random

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Federated Using Cuda")
torch.manual_seed(random.random() * 100)

class Local_Model:
    def __init__(self, network_architecture, train_loader, validation_loader, test_loader, N_partitions, learning_rate):
        with open('config.json') as config_file:
            config = json.load(config_file)
            DATA_PARALLEL = config['machine_learning']['DATA_PARALLEL']
        
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.model = network_architecture().to(device=DEVICE)
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        if DATA_PARALLEL and torch.cuda.is_available() and (torch.cuda.device_count() > 1):
            print("Using Data Parallel")
            self.model = nn.DataParallel(self.model)

        nn_architectures.parameter_init(self.model)

    def transfer_param_to_model(self, param):
        for w_model, w_param in zip(self.model.parameters(), param):
            w_model.data = w_param.data

    def train(self):
        self.model.train()
        for batch_idx, (train_x, train_y) in enumerate(self.train_loader):
            train_x = train_x.to(DEVICE)
            train_y = train_y.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(train_x)
            loss = self.loss_fn(outputs, train_y)
            loss.backward()
            self.optimizer.step()

        return (loss, self.model.parameters())

    def get_accuracy(self):
        acc = evaluate.accuracy(self.model, self.test_loader)
        return acc

    def get_validation_accuracy(self):
        return evaluate.accuracy(self.model, self.validation_loader)

    def get_selective_aggregation_vector(self):
        return evaluate.selective_aggregating_vector(self.model, self.validation_loader)

    def get_loss(self):
        loss = evaluate.loss(self.model, self.train_loader, self.loss_fn)
        return loss

    def get_model(self):
        return self.model
    
class Aggregated_Model:
    def __init__(self, network_architecture, N_partitions):
        self.model = network_architecture().to(device=DEVICE)
        self.loss_fn = nn.NLLLoss()

        nn_architectures.parameter_init(self.model)

    def aggregate_central(self, local_params):
        N_partitions = len(local_params)

        for w_global in self.model.parameters():
            w_global.data = w_global.data * 0
        for local_param in local_params:
            for w_global, w_local in zip(self.model.parameters(), local_param):
                    w_global.data += w_local.data / N_partitions
    
        return self.model.parameters()

    def get_params(self):
        return self.model.parameters()

    def get_model(self):
        return self.model

    def get_test_loader(self):
        return self.test_loader
        
def set_device(device):
    global DEVICE
    if torch.cuda.is_available():
        DEVICE = torch.device(device)

def get_device():
    global DEVICE
    return DEVICE