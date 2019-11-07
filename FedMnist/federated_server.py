import federated
from model import nn_architectures, data_loader

import requests
import sys

def fed_avg_done():
    network_architecture = nn_architectures.NetFC
    get_train_loader = data_loader.get_empty_train_loader
    get_test_loader = data_loader.get_unified_test_loader
    N_partitions = 3

    local_nets = [federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions, i) for i in range(N_partitions)]
    global_net = federated.Aggregated_Model(network_architecture, get_test_loader, N_partitions)

    payload = {'status': "success"}
    r = requests.get('http://localhost:3000/fed_avg_done', params=payload)

fed_avg_done()
