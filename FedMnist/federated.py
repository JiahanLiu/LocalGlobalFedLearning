import evaluate
from model import nn_architectures

import torch
import torch.nn as nn
import torch.optim as optim

import configparser
import random

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Federated Using Cuda")
torch.manual_seed(random.random() * 100)

class Local_Model:
    def __init__(self, network_architecture, get_train_loader, get_test_loader, N_partitions, node_id):
        config = configparser.RawConfigParser()
        config.read('config.cfg')
        FC_LEARNING_RATE = float(config['DEFAULT']['FC_LEARNING_RATE'])

        train_loaders, validation_loader = get_train_loader(N_partitions)
        test_loaders = get_test_loader(N_partitions)
        
        self.test_loader = test_loaders
        if(len(test_loaders) > 1):
            self.test_loader = test_loaders[node_id]
        self.train_loader = train_loaders
        if(N_partitions != 0):
            self.train_loader = train_loaders[node_id] 
        self.model = network_architecture().to(device=DEVICE)
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=FC_LEARNING_RATE)

        nn_architectures.xavier_init(self.model)

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

    def get_loss(self):
        loss = evaluate.loss(self.model, self.train_loader, self.loss_fn)
        return loss

    def get_model(self):
        return self.model
    
class Aggregated_Model:
    def __init__(self, network_architecture, get_test_loader, N_partitions):
        self.test_loader = get_test_loader(N_partitions)
        self.model = network_architecture().to(device=DEVICE)
        self.loss_fn = nn.NLLLoss()

        nn_architectures.xavier_init(self.model)

    def aggregate_central(self, local_params):
        N_partitions = len(local_params)

        for w_global in self.model.parameters():
            w_global.data = w_global.data * 0
        for local_param in local_params:
            for w_global, w_local in zip(self.model.parameters(), local_param):
                    w_global.data += w_local.data / N_partitions
    
        return self.model.parameters()

    def get_accuracy(self):
        acc = evaluate.accuracy(model=self.model, data_loader=self.test_loader)
        return acc

    def get_params(self):
        return self.model.parameters()

    def get_test_loader(self):
        return self.test_loader


        
