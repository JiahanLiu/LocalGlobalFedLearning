import torch
from torchvision import datasets, transforms

import math
import os

BATCH_SIZE_TRAIN = 256
VALIDATION_SIZE = 10000
BATCH_SIZE_TEST = 10000

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

def get_train_datasets():
    train_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=True, 
        transform=transforms.ToTensor(),
        download=True)
        
    return train_dataset

def get_test_datasets():
    test_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=False, 
        transform=transforms.ToTensor(),
        download=True)
    
    return test_dataset

def get_unified_train_loader():
    train_dataset = get_train_datasets()

    train_set = PartitionedDataset()
    validation_set = PartitionedDataset()

    partition_size = len(train_dataset) - VALIDATION_SIZE

    index = 0
    for j in range(VALIDATION_SIZE):
        item = train_dataset.__getitem__(index)
        index = index + 1
        validation_set.__add__(item)
    for j in range(partition_size):
        item = train_dataset.__getitem__(index)
        index = index + 1
        train_set.__add__(item)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

    return train_loader, validation_loader

def get_unified_test_loader(N_partitions):
    test_dataset = get_test_datasets()

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

    return test_loader

def get_random_partitioned_train_loaders(N_partitions):
    train_dataset = get_train_datasets()

    validation_set = PartitionedDataset()
    paritioned_train_sets = [PartitionedDataset() for n in range(N_partitions)]

    partition_size = math.floor((len(train_dataset) - VALIDATION_SIZE)/ N_partitions)

    index = 0
    for j in range(VALIDATION_SIZE):
        item = train_dataset.__getitem__(index)
        index = index + 1
        validation_set.__add__(item)
    for i in range(N_partitions):
        for j in range(partition_size):
            item = train_dataset.__getitem__(index)
            index = index + 1
            paritioned_train_sets[i].__add__(item)

    train_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True) for dataset in paritioned_train_sets]
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

    return (train_loaders, validation_loader)

def get_unbalanced_partitioned_train_loaders(N_partitions):
    train_dataset = get_train_datasets()
    validation_set = PartitionedDataset()
    paritioned_train_sets = [PartitionedDataset() for n in range(N_partitions)]

    total_size = math.floor((len(train_dataset) - VALIDATION_SIZE))
    partition_cutoffs = [(math.floor(10 / N_partitions) * i) for i in range(0, N_partitions)]

    index = 0
    for j in range(VALIDATION_SIZE):
        item = train_dataset.__getitem__(index)
        index = index + 1 
        validation_set.__add__(item)
    for i in range(total_size):
        item = train_dataset.__getitem__(index)
        index = index + 1
        for i in range(N_partitions-1):
            if ((partition_cutoffs[i] <= item[1]) and (item[1] < partition_cutoffs[i+1])):
                paritioned_train_sets[i].__add__(item)
        if (partition_cutoffs[N_partitions-1] <= item[1]):
            paritioned_train_sets[N_partitions-1].__add__(item)

    train_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True) for dataset in paritioned_train_sets]
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

    return (train_loaders, validation_loader)

def get_unbalanced_partitioned_test_loaders(N_partitions):
    test_dataset = get_test_datasets()
    
    paritioned_test_sets = [PartitionedDataset() for n in range(N_partitions)]

    total_test_size = math.floor(len(test_dataset))
    partition_cutoffs = [(math.floor(10 / N_partitions) * i) for i in range(0, N_partitions)]

    index = 0
    N_test_partitions = [0 for i in range(N_partitions)]
    for i in range(total_test_size):
        item = test_dataset.__getitem__(index)
        index = index + 1
        for i in range(N_partitions-1):
            if ((partition_cutoffs[i] <= item[1]) and (item[1] < partition_cutoffs[i+1])):
                paritioned_test_sets[i].__add__(item)
                N_test_partitions[i] = N_test_partitions[i] + 1
        if (partition_cutoffs[N_partitions-1] <= item[1]):
            paritioned_test_sets[N_partitions-1].__add__(item)
            N_test_partitions[N_partitions-1] = N_test_partitions[N_partitions-1] + 1

    test_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE_TEST, shuffle=False) for dataset, size in zip(paritioned_test_sets, N_test_partitions)]
    
    return test_loaders

def get_semibalanced_partitioned_train_loader_closure(percentage):

    def get_semibalanced_partitioned_train_loader(N_partitions):
        train_dataset = get_train_datasets()
        validation_set = PartitionedDataset()
        paritioned_train_sets = [PartitionedDataset() for n in range(N_partitions)]

        total_size = math.floor((len(train_dataset) - VALIDATION_SIZE))
        balanced_size = math.floor((total_size * percentage) / 100)
        balanced_partition_size = math.floor(balanced_size / N_partitions)
        unbalanced_size = total_size - balanced_size
        partition_cutoffs = [(math.floor(10 / N_partitions) * i) for i in range(0, N_partitions)]

        print("here2: " + str(balanced_size))

        index = 0
        for j in range(VALIDATION_SIZE):
            item = train_dataset.__getitem__(index)
            index = index + 1 
            validation_set.__add__(item)
        for i in range(N_partitions):
            for j in range(balanced_partition_size):
                item = train_dataset.__getitem__(index)
                index = index + 1
                paritioned_train_sets[i].__add__(item)

        print("here1: " + str(index))

        for i in range(unbalanced_size):
            item = train_dataset.__getitem__(index)
            index = index + 1
            for i in range(N_partitions-1):
                if ((partition_cutoffs[i] <= item[1]) and (item[1] < partition_cutoffs[i+1])):
                    paritioned_train_sets[i].__add__(item)
            if (partition_cutoffs[N_partitions-1] <= item[1]):
                paritioned_train_sets[N_partitions-1].__add__(item)

        train_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True) for dataset in paritioned_train_sets]
        validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

        return (train_loaders, validation_loader)

    return get_semibalanced_partitioned_train_loader
