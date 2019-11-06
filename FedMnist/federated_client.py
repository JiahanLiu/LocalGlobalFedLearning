import federated
from model import nn_architectures, data_loader

import torch

import configparser
import os.path
import requests

SERVER_URL_BASE = None
UPLOAD_URL = None
DOWNLOAD_URL = None

PARAM_FILE_DIR = "client/param_files/"

def upload_local_param(upload_url, file_path, file_name):
    with open(file_path, 'rb') as file:
        files = {'file':(file_name, file)}
        r = requests.post(upload_url, files=files)
        print(r)
        print("Local Params Uploaded")

def download_global_param(download_url, file_path, file_name):
    payload = {'filename': file_name}
    r = requests.get(download_url, params=payload, allow_redirects=True)
    open(file_path, 'wb').write(r.content)
    print("File Downloaded")

def save_model_to_file(model, file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
    torch.save(model.state_dict(), file_path)

def load_model_from_file(model, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)

def federated_local(network_architecture, get_train_loader, get_test_loader, n_epochs):
    local_param_file_name = "node0_local_param.pt"
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    local_param_file_path = os.path.join(pwd_path, PARAM_FILE_DIR, local_param_file_name)
    global_param_file_name = "global_param.pt"
    global_param_file_path = os.path.join(pwd_path, PARAM_FILE_DIR, global_param_file_name)

    net = federated.Local_Model(network_architecture, get_train_loader, get_test_loader, N_partitions=1, node_id=0)

    for epoch in range(n_epochs):
        (loss, local_param) = net.train()
        save_model_to_file(net.get_model(), local_param_file_path)
        upload_local_param(UPLOAD_URL, local_param_file_path, local_param_file_name)
        download_global_param(DOWNLOAD_URL, global_param_file_path, global_param_file_name)
        load_model_from_file(net.get_model(), global_param_file_path)
        updated_param = net.get_model().parameters()
        net.transfer_param_to_model(updated_param)
        acc = net.get_accuracy()
        print("Epoch: " + str(epoch) + " | Accuracy: " + str(acc))

def main():
    global SERVER_URL_BASE
    global UPLOAD_URL
    global DOWNLOAD_URL
    config = configparser.RawConfigParser()
    config.read('config.cfg')
    SERVER_URL_BASE = config['CLIENT']['SERVER_URL_BASE']
    UPLOAD_URL = SERVER_URL_BASE + config['CLIENT']['UPLOAD_ROUTE']
    DOWNLOAD_URL = SERVER_URL_BASE + config['CLIENT']['DOWNLOAD_ROUTE']
    N_EPOCHS = int(config['DEFAULT']['N_EPOCHS'])

    federated_local(nn_architectures.NetFC, data_loader.get_random_partitioned_train_loaders, data_loader.get_unified_test_loader, N_EPOCHS)

if __name__ == "__main__":
    main()
