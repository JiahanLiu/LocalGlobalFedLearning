import torch
from torchvision import datasets, transforms

import math
import os

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

def fill_set_straight(train_dataset, target_set, target_set_size, index):
    for j in range(target_set_size):
        item = train_dataset.__getitem__(index)
        index = index + 1
        target_set.__add__(item)

    return index

def fill_random_set(train_dataset, paritioned_train_sets, partition_size, N_partitions, index):
    for j in range(partition_size):
        for i in range(N_partitions):
            item = train_dataset.__getitem__(index)
            index = index + 1
            paritioned_train_sets[i].__add__(item)
    
    return index

def fill_unbalanced_set(train_dataset, paritioned_target_sets, total_size, N_partitions, partition_cutoffs, index):
    for i in range(total_size):
        item = train_dataset.__getitem__(index)
        index = index + 1
        for i in range(N_partitions-1):
            if ((partition_cutoffs[i] <= item[1]) and (item[1] < partition_cutoffs[i+1])):
                paritioned_target_sets[i].__add__(item)
        if (partition_cutoffs[N_partitions-1] <= item[1]):
            paritioned_target_sets[N_partitions-1].__add__(item)

    return index

def fill_selective_unbalanced_set(train_dataset, paritioned_target_sets, total_size, N_partitions, partition_cutoffs, similar_count, index):
    modular_index = [0 for i in range(similar_count)]
    for i in range(total_size):
        item = train_dataset.__getitem__(index)
        index = index + 1
        for i in range(int(N_partitions/similar_count)-1):
            if ((partition_cutoffs[i] <= item[1]) and (item[1] < partition_cutoffs[i+1])):
                target_index = similar_count * i + modular_index[i]
                paritioned_target_sets[target_index].__add__(item)
                modular_index[i] = (modular_index[i] + 1) % similar_count
        if (partition_cutoffs[int(N_partitions/similar_count)-1] <= item[1]):
            target_index = similar_count * (int(N_partitions/similar_count)-1) + modular_index[similar_count-1]
            paritioned_target_sets[target_index].__add__(item)
            modular_index[similar_count-1] = (modular_index[similar_count-1] + 1) % similar_count

    return index

def get_unified_train_loader_closure(batch_size):
    def get_unified_train_loader(polymorphism_filler):
        train_dataset = get_train_datasets()

        train_set = PartitionedDataset()
        validation_set = PartitionedDataset()

        partition_size = len(train_dataset) - VALIDATION_SIZE

        index = 0
        index = fill_set_straight(train_dataset, validation_set, VALIDATION_SIZE, index)
        for j in range(partition_size):
            item = train_dataset.__getitem__(index)
            index = index + 1
            train_set.__add__(item)

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

        return train_loader, validation_loader
    return get_unified_train_loader

def get_unified_test_loader(polymorphism_filler):
    test_dataset = get_test_datasets()
    total_size = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=total_size, shuffle=False)

    return test_loader

def get_random_partitioned_train_loaders_closure(batch_size):
    def get_random_partitioned_train_loaders(N_partitions):
        partitioned_train_loaders = get_semibalanced_partitioned_train_loaders_closure(100, batch_size)
        (train_loaders, validation_loader) = partitioned_train_loaders(N_partitions)

        return (train_loaders, validation_loader)
    return get_random_partitioned_train_loaders

def get_random_partitioned_test_loaders(N_partitions):
    get_semibalanced_partitioned_test_loaders = get_semibalanced_partitioned_test_loaders_closure(100)
    test_loaders = get_semibalanced_partitioned_test_loaders(N_partitions)

    return test_loaders

def get_unbalanced_partitioned_train_loaders_closure(batch_size):
    def get_unbalanced_partitioned_train_loaders(N_partitions):
        partitioned_train_loaders = get_semibalanced_partitioned_train_loaders_closure(0, batch_size)
        (train_loaders, validation_loader) = partitioned_train_loaders(N_partitions)

        return (train_loaders, validation_loader)
    return get_unbalanced_partitioned_train_loaders

def get_unbalanced_partitioned_test_loaders(N_partitions):
    get_semibalanced_partitioned_test_loaders = get_semibalanced_partitioned_test_loaders_closure(0)
    test_loaders = get_semibalanced_partitioned_test_loaders(N_partitions)

    return test_loaders

def get_semibalanced_partitioned_train_loaders_closure(percentage_balanced, batch_size):
    def get_semibalanced_partitioned_train_loaders(N_partitions):
        train_dataset = get_train_datasets()

        partitioned_validation_sets = [PartitionedDataset() for n in range(N_partitions)]
        paritioned_train_sets = [PartitionedDataset() for n in range(N_partitions)]

        validation_balanced_size = math.floor((VALIDATION_SIZE * percentage_balanced) / 100)
        validation_balanced_partition_size = math.floor(validation_balanced_size / N_partitions)
        validation_unbalanced_size = VALIDATION_SIZE - validation_balanced_size

        total_size = math.floor((len(train_dataset) - VALIDATION_SIZE))
        balanced_size = math.floor((total_size * percentage_balanced) / 100)
        balanced_partition_size = math.floor(balanced_size / N_partitions)
        unbalanced_size = total_size - balanced_size
        partition_cutoffs = [(math.floor(10 / N_partitions) * i) for i in range(0, N_partitions)]

        index = 0
        index = fill_random_set(train_dataset, partitioned_validation_sets, validation_balanced_partition_size, N_partitions, index)
        index = fill_unbalanced_set(train_dataset, partitioned_validation_sets, validation_unbalanced_size, N_partitions, partition_cutoffs, index)
        index = fill_random_set(train_dataset, paritioned_train_sets, balanced_partition_size, N_partitions, index)
        index = fill_unbalanced_set(train_dataset, paritioned_train_sets, unbalanced_size, N_partitions, partition_cutoffs, index)

        train_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) for dataset in paritioned_train_sets]
        validation_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=VALIDATION_SIZE, shuffle=False) for dataset in partitioned_validation_sets]

        return (train_loaders, validation_loaders)

    return get_semibalanced_partitioned_train_loaders

def get_selective_aggregation_train_loaders_closure(percentage_balanced, batch_size, similar_count):
    def get_selective_aggregation_train_loaders(N_partitions): #N_partitions is the total num of nodes. 3 dist x 3 similar = 9 N_partition
        train_dataset = get_train_datasets()

        partitioned_validation_sets = [PartitionedDataset() for n in range(N_partitions)] # all sets are the same 10k to calculate node distance
        paritioned_train_sets = [PartitionedDataset() for n in range(N_partitions)]

        validation_balanced_partition_size = VALIDATION_SIZE

        total_size = math.floor((len(train_dataset) - VALIDATION_SIZE))
        balanced_size = math.floor((total_size * percentage_balanced) / 100)
        balanced_partition_size = math.floor(balanced_size / N_partitions)
        unbalanced_size = total_size - balanced_size
        partition_cutoffs = [(math.floor(10 / (int(N_partitions/similar_count))) * i) for i in range(0, int(N_partitions/similar_count))]

        index = 0
        for i in range(N_partitions):
            index = 0
            index = fill_set_straight(train_dataset, partitioned_validation_sets[i], validation_balanced_partition_size, index)
        index = fill_random_set(train_dataset, paritioned_train_sets, balanced_partition_size, N_partitions, index)
        index = fill_selective_unbalanced_set(train_dataset, paritioned_train_sets, unbalanced_size, N_partitions, partition_cutoffs, similar_count, index)

        train_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) for dataset in paritioned_train_sets]
        validation_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=VALIDATION_SIZE, shuffle=False) for dataset in partitioned_validation_sets]

        return (train_loaders, validation_loaders)

    return get_selective_aggregation_train_loaders

def get_semibalanced_partitioned_test_loaders_closure(percentage_balanced):
    def get_semibalanced_partitioned_test_loaders(N_partitions):
        test_dataset = get_test_datasets()

        paritioned_test_sets = [PartitionedDataset() for n in range(N_partitions)]

        total_size = len(test_dataset)
        balanced_size = math.floor((total_size * percentage_balanced) / 100)
        balanced_partition_size = math.floor(balanced_size / N_partitions)
        unbalanced_size = total_size - balanced_size
        partition_cutoffs = [(math.floor(10 / N_partitions) * i) for i in range(0, N_partitions)]

        index = 0
        index = fill_random_set(test_dataset, paritioned_test_sets, balanced_partition_size, N_partitions, index)
        index = fill_unbalanced_set(test_dataset, paritioned_test_sets, unbalanced_size, N_partitions, partition_cutoffs, index)

        test_loaders = [torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False) for dataset in paritioned_test_sets]

        return test_loaders
    return get_semibalanced_partitioned_test_loaders
