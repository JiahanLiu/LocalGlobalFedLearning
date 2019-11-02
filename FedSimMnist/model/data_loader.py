import torch
from torchvision import datasets, transforms

import math
import os

BATCH_SIZE_TRAIN = 256
VALIDATION_SIZE = 10000
BATCH_SIZE_TEST = 1

DIRPATH = os.getcwd()
DATAPATH = DIRPATH + '/data/'

class PartitionedDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
        
    def __add__(self, other):
        return self.dataset.append(other)

def get_datasets():
    train_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=True, 
        transform=transforms.ToTensor(),
        download=True)
    test_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=False, 
        transform=transforms.ToTensor(),
        download=True)

    return train_dataset, test_dataset

def get_loaders():
    (train_dataset, test_dataset) = get_datasets()

    validation_set = PartitionedDataset()
    train_set_only = PartitionedDataset()

    index = 0
    partition_size = (len(train_dataset) - VALIDATION_SIZE)


    for j in range(VALIDATION_SIZE):
        item = train_dataset.__getitem__(index)
        index = index + 1
        validation_set.__add__(item)

    for j in range(partition_size):
        item = train_dataset.__getitem__(index)
        index = index + 1
        train_set_only.__add__(item)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

    return train_loader, validation_loader, test_loader

def get_random_partitioned_loaders(N_partitions):
    (train_dataset, test_dataset) = get_datasets()

    paritioned_train_sets = [PartitionedDataset() for n in range(N_partitions)]

    index = 0
    partition_size = math.floor(len(train_dataset) / N_partitions)

    for i in range(N_partitions):
        for j in range(partition_size):
            item = train_dataset.__getitem__(index)
            index = index + 1
            paritioned_train_sets[i].__add__(item)

    train_loaders = [torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True) for train_dataset in paritioned_train_sets]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

    return train_loaders, test_loader