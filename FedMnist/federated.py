
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

    def __transfer_param_to_model(self, model, param):
        for w_model, w_param in zip(model.parameters(), param):
            w_model.data = w_param.data

    def __train_epoch(self, model, train_loader, loss_fn, optimizer):
        model.train()
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.to(DEVICE)
            train_y = train_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(train_x)
            loss = loss_fn(outputs, train_y)
            loss.backward()
            optimizer.step()

        return (loss, model.parameters())

    def train(self, global_model):
        self.__transfer_param_to_model(model=self.model, param=global_model.parameters())
        # print(len(self.train_loader))
        (loss, local_param) = self.__train_epoch(model=self.model, train_loader=self.train_loader, loss_fn=self.loss_fn, optimizer=self.optimizer)

        return (loss, local_param)