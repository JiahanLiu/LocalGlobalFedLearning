import evaluate

import torch
import torch.nn as nn
import torch.optim as optim

import configparser

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Federated Using Cuda")

class Federated_Local:
    def __init__(self, network_architecture, get_train_loader, N_partitions, node_id):
        config = configparser.RawConfigParser()
        config.read('config.cfg')
        FC_LEARNING_RATE = float(config['DEFAULT']['FC_LEARNING_RATE'])

        train_loaders, validation_loader = get_train_loader(N_partitions)
        self.train_loader = train_loaders[node_id] 
        self.model = network_architecture().to(device=DEVICE)
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=FC_LEARNING_RATE)

    def __transfer_param_to_model(self, param):
        for w_model, w_param in zip(self.model.parameters(), param):
            w_model.data = w_param.data

    def __train_epoch(self):
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

    def train(self, global_model):
        self.__transfer_param_to_model(param=global_model.parameters())
        (loss, local_param) = self.__train_epoch()

        return (loss, local_param)

class Federated_Global:
    def __init__(self, network_architecture, get_test_loader, N_partitions):
        self.test_loader = get_test_loader(N_partitions)
        self.model = network_architecture().to(device=DEVICE)
        self.loss_fn = nn.NLLLoss()

        self.__init_weights(self.model)

    def __init_weights(self, model):
        if type(model) == nn.Linear:
            torch.nn.init.xavier_uniform(model.weight)
            model.bias.data.fill_(0.01)

    def aggregate_central(self, local_params):
        N_partitions = len(local_params)

        for w_global in self.model.parameters():
            w_global.data = w_global.data * 0
        for local_param in local_params:
            for w_global, w_local in zip(self.model.parameters(), local_param):
                    w_global.data += w_local.data / N_partitions
    
        return self.model.parameters()

    def get_accuracy():
        acc = evaluate.accuracy(model=self.model, data_loader=self.test_loader)
        return acc

    def get_model(self):
        return self.model

    def get_test_loader(self):
        return self.test_loader




        
