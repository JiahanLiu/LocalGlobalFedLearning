import torch
from torchvision import datasets, transforms

import math
import os

BATCH_SIZE_TRAIN = 256
VALIDATION_SIZE = 10000

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

def fill_validation_set(train_dataset, validation_set, index):
    for j in range(VALIDATION_SIZE):
        item = train_dataset.__getitem__(index)
        index = index + 1
        validation_set.__add__(item)

    return index

def fill_random_set(train_dataset, paritioned_train_sets, partition_size, N_partitions, index):
    for j in range(partition_size):
        for i in range(N_partitions):
            item = train_dataset.__getitem__(index)
            index = index + 1
            paritioned_train_sets[i].__add__(item)
    
    return index

def fill_unbalanced_set(train_dataset, paritioned_train_sets, total_size, N_partitions, partition_cutoffs, index):
    for i in range(total_size):
        item = train_dataset.__getitem__(index)
        index = index + 1
        for i in range(N_partitions-1):
            if ((partition_cutoffs[i] <= item[1]) and (item[1] < partition_cutoffs[i+1])):
                paritioned_train_sets[i].__add__(item)
        if (partition_cutoffs[N_partitions-1] <= item[1]):
            paritioned_train_sets[N_partitions-1].__add__(item)

    return index

# def get_empty_train_loader(polymorphism_filler):
#     train_set = PartitionedDataset()
#     validation_set = PartitionedDataset()

#     train_set.__add__(torch.tensor([0]))
#     validation_set.__add__(torch.tensor([0]))

#     train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
#     validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

#     return train_loader, validation_loader

def get_unified_train_loader(polymorphism_filler):
    train_dataset = get_train_datasets()

    train_set = PartitionedDataset()
    validation_set = PartitionedDataset()

    partition_size = len(train_dataset) - VALIDATION_SIZE

    index = 0
    index = fill_validation_set(train_dataset, validation_set, index)
    for j in range(partition_size):
        item = train_dataset.__getitem__(index)
        index = index + 1
        train_set.__add__(item)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

    return train_loader, validation_loader

def get_unified_test_loader(polymorphism_filler):
    test_dataset = get_test_datasets()
    total_size = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=total_size, shuffle=False)

    return test_loader

def get_random_partitioned_train_loaders(N_partitions):
    random_partitioned_train_loaders = get_semibalanced_partitioned_train_loaders_closure(100)
    (train_loaders, validation_loader) = random_partitioned_train_loaders(N_partitions)

    return (train_loaders, validation_loader)

def get_unbalanced_partitioned_train_loaders(N_partitions):
    random_partitioned_train_loaders = get_semibalanced_partitioned_train_loaders_closure(0)
    (train_loaders, validation_loader) = random_partitioned_train_loaders(N_partitions)

    return (train_loaders, validation_loader)

def get_unbalanced_partitioned_test_loaders(N_partitions):
    test_dataset = get_test_datasets()
    
    paritioned_test_sets = [PartitionedDataset() for n in range(N_partitions)]

    total_size = len(test_dataset)
    partition_cutoffs = [(math.floor(10 / N_partitions) * i) for i in range(0, N_partitions)]

    index = 0
    index = fill_unbalanced_set(test_dataset, paritioned_test_sets, total_size, N_partitions, partition_cutoffs, index)
    
    test_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False) for dataset in paritioned_test_sets]

    return test_loaders

def get_semibalanced_partitioned_train_loaders_closure(percentage_balanced):

    def get_semibalanced_partitioned_train_loaders(N_partitions):
        train_dataset = get_train_datasets()

        validation_set = PartitionedDataset()
        paritioned_train_sets = [PartitionedDataset() for n in range(N_partitions)]

        total_size = math.floor((len(train_dataset) - VALIDATION_SIZE))
        balanced_size = math.floor((total_size * percentage_balanced) / 100)
        balanced_partition_size = math.floor(balanced_size / N_partitions)
        unbalanced_size = total_size - balanced_size
        partition_cutoffs = [(math.floor(10 / N_partitions) * i) for i in range(0, N_partitions)]

        index = 0
        index = fill_validation_set(train_dataset, validation_set, index)
        index = fill_random_set(train_dataset, paritioned_train_sets, balanced_partition_size, N_partitions, index)
        index = fill_unbalanced_set(train_dataset, paritioned_train_sets, unbalanced_size, N_partitions, partition_cutoffs, index)

        train_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True) for dataset in paritioned_train_sets]
        validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

        return (train_loaders, validation_loader)

    return get_semibalanced_partitioned_train_loaders
