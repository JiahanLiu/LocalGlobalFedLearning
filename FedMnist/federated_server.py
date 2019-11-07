import federated
from model import nn_architectures, data_loader
import util

import torch

import os
import requests
import sys

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Federated Using Cuda")

PARAM_FILE_DIR = "server/param_files/"

def build_local_param_path(node_n):
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    local_param_file_name = "node" + str(node_n) + "_local_param.pt"
    local_param_file_path = os.path.join(pwd_path, PARAM_FILE_DIR, local_param_file_name)
    return local_param_file_path

def build_global_param_path():
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    global_param_file_name = "global_param.pt"
    global_param_file_path = os.path.join(pwd_path, PARAM_FILE_DIR, global_param_file_name)
    return global_param_file_path

def fed_avg():
    network_architecture = nn_architectures.NetFC
    get_test_loader = data_loader.get_unified_test_loader
    N_partitions = 3

    local_nets = [network_architecture() for i in range(N_partitions)]
    global_net = federated.Aggregated_Model(network_architecture, get_test_loader, N_partitions)

    for i in range(N_partitions):
        util.load_model_from_file(local_nets[i], build_local_param_path(i), DEVICE)
    local_params = [local_nets[i].parameters() for i in range(N_partitions)]

    global_net.aggregate_central(local_params)
    util.save_model_to_file(global_net.get_model(), build_global_param_path())

    payload = {'status': "success"}
    r = requests.get('http://localhost:3000/fed_avg_done', params=payload)

def main():
    fed_avg()

if __name__ == "__main__":
    main()