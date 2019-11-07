import federated
from model import nn_architectures, data_loader
import util

import torch

import configparser
import json
import os.path
import requests
import sys
import time

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Federated Using Cuda")

SERVER_URL_BASE = None
UPLOAD_URL = None
DOWNLOAD_URL = None
QUERY_FED_AVG_DONE_URL = None
QUERY_GLOBAL_SYNC_DONE_URL = None

PARAM_FILE_DIR = "client/param_files/"

def upload_local_param(upload_url, file_path, file_name, node_n):
    with open(file_path, 'rb') as file:
        files = {'file':(file_name, file)}
        payload = {'node_n' : node_n}
        r = requests.post(upload_url, files=files, params=payload)
        print(r)
        print("Local Params Uploaded")

def download_global_param(download_url, file_path, file_name, node_n):
    payload = {'filename': file_name, 'node_n': node_n}
    r = requests.get(download_url, params=payload, allow_redirects=True)
    open(file_path, 'wb').write(r.content)
    print("File Downloaded")

def wait_fed_avg_done():
    r = requests.get(QUERY_FED_AVG_DONE_URL)
    r_json = r.json()
    while(r_json['status'] == 0):
        r = requests.get(QUERY_FED_AVG_DONE_URL)
        r_json = r.json()
        print("Waiting...")
        time.sleep(1)

    return
    
def wait_global_sync_done():
    r = requests.get(QUERY_GLOBAL_SYNC_DONE_URL)
    r_json = r.json()
    while(r_json['status'] == 0):
        r = requests.get(QUERY_GLOBAL_SYNC_DONE_URL)
        r_json = r.json()
        print("Waiting...")
        time.sleep(1)
    return

def federated_local(network_architecture, get_train_loader, get_test_loader, n_epochs, node_n):
    local_param_file_name = "node" + str(node_n) + "_local_param.pt"
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    local_param_file_path = os.path.join(pwd_path, PARAM_FILE_DIR, local_param_file_name)
    global_param_file_name = "global_param.pt"
    global_param_file_path = os.path.join(pwd_path, PARAM_FILE_DIR, global_param_file_name)

    net = federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions=1, node_id=0)
    global_net = network_architecture().to(device=DEVICE)

    for epoch in range(n_epochs):
        wait_global_sync_done()
        (loss, local_param) = net.train()
        util.save_model_to_file(net.get_model(), local_param_file_path)
        upload_local_param(UPLOAD_URL, local_param_file_path, local_param_file_name, node_n)
        
        wait_fed_avg_done()
        download_global_param(DOWNLOAD_URL, global_param_file_path, global_param_file_name, node_n)
        util.load_model_from_file(global_net, global_param_file_path, DEVICE)
        updated_param = global_net.parameters()
        net.transfer_param_to_model(updated_param)

        acc = net.get_accuracy()
        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc))

def main():
    if(len(sys.argv) < 2):
        print("Error: Need arg[1] for node_n - the node number.")
    node_n = sys.argv[1]

    global SERVER_URL_BASE
    global UPLOAD_URL
    global DOWNLOAD_URL
    global QUERY_GLOBAL_SYNC_DONE_URL
    global QUERY_FED_AVG_DONE_URL
    config = configparser.RawConfigParser()
    config.read('config.cfg')
    SERVER_URL_BASE = config['CLIENT']['SERVER_URL_BASE']
    UPLOAD_URL = SERVER_URL_BASE + config['CLIENT']['UPLOAD_ROUTE']
    DOWNLOAD_URL = SERVER_URL_BASE + config['CLIENT']['DOWNLOAD_ROUTE']
    QUERY_GLOBAL_SYNC_DONE_URL = SERVER_URL_BASE + config['CLIENT']['QUERY_GLOBAL_SYNC_DONE_ROUTE']
    QUERY_FED_AVG_DONE_URL = SERVER_URL_BASE + config['CLIENT']['QUERY_FED_AVG_DONE_ROUTE']
    N_EPOCHS = int(config['DEFAULT']['N_EPOCHS'])

    federated_local(nn_architectures.NetFC, data_loader.get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS, node_n)

if __name__ == "__main__":
    main()
