# ![MPC](https://github.com/UT-MPC/LocalGlobalFedLearning/raw/master/Papers/scratch/MPC.png)

[![Build Status](https://travis-ci.org/sindresorhus/pageres.svg?branch=master)](https://travis-ci.org/sindresorhus/pageres) [![Coverage Status](https://coveralls.io/repos/sindresorhus/pageres/badge.svg?branch=master)](https://coveralls.io/r/sindresorhus/pageres?branch=master) [![XO code style](https://img.shields.io/badge/code_style-XO-5ed9c7.svg)](https://github.com/xojs/xo)

Federated Learning Research Library for [Mobile and Pervasive Computing Group](http://mpc.ece.utexas.edu/).

## Workflow

| Step        | Goal           | File  |
| ------------- |:-------------:| -----:|
| 1     | Dataloader | `/FedMnist/model/data_loader.py` |
| 2     | Neural Network Architectures | `/FedMnist/model/nn_architectures.py` |
| 3     | Training Modes (Currently Centralized, Local, Federated, Selective Fed) | `/FedMnist/train.py` |
| 4     | Search Hyperparameters | `/FedMnist/search_hyperparameter.py` |
| 5     | Use Hyperparameter Results to Manually Update Config File | `/FedMnist/config.json` |
| 6     | Generate Results | `/FedMnist/synthesize_results.py` |
| 7     | Deployment Server Networking | `/FedMnist/Server/app.js` |
| 8     | Deployment Server | `/FedMnist/federated_server.py` |
| 9     | Deployment Client | `/FedMnist/federated_client.py` |
| 10     | Graph Results (More Nodes Branch) | `/FedMnist/graph.py` |

## Dependencies
The federated library is built on Pytorch on Python3. The deployed server uses node.js. 

##### Python3 Dependencies
The FedMnist programs dependencies can be found in this [Docker](https://hub.docker.com/r/vastai/pytorch/). Below are the install packages that which the Production branch uses on a Vast.AI server.
```
Python 3.7.1
Package          Version   
---------------- ----------
asn1crypto       0.24.0    
backcall         0.1.0     
certifi          2018.11.29
cffi             1.11.5    
chardet          3.0.4     
conda            4.5.12    
cryptography     2.4.2     
cycler           0.10.0    
Cython           0.29.2    
decorator        4.3.0     
idna             2.7       
ipython          7.2.0     
ipython-genutils 0.2.0     
jedi             0.13.2    
kiwisolver       1.1.0     
matplotlib       3.1.2     
mkl-fft          1.0.6     
mkl-random       1.0.2     
numpy            1.15.4    
olefile          0.46      
parso            0.3.1     
pexpect          4.6.0     
pickleshare      0.7.5     
Pillow           5.3.0     
pip              10.0.1    
prompt-toolkit   2.0.7     
ptyprocess       0.6.0     
pycosat          0.6.3     
pycparser        2.18      
Pygments         2.3.1     
pyOpenSSL        18.0.0    
pyparsing        2.4.5     
PySocks          1.6.8     
python-dateutil  2.8.1     
PyYAML           3.13      
requests         2.19.1    
ruamel-yaml      0.15.46   
scipy            1.1.0     
setuptools       40.2.0    
six              1.11.0    
torch            1.0.0     
torchvision      0.2.1     
traitlets        4.3.2     
typing           3.6.4     
urllib3          1.23      
wcwidth          0.1.7     
wheel            0.31.1    
```
##### Node.js Dependencies
```
    "express": "^4.17.1",
    "multer": "^1.4.2"
```

## Software Design
The software library is built using the Pytorch, a popular python library for machine learning. Federated.py contains all the code to perform a single round of federated
learning in both the experiments as well as the deployed test setup. It allows creation of a local and global classes to represent or be deployed on local nodes and the
aggregating server respectively. The data loaders and model architecture are congurable and passed in as parameters to allow for 
exibility in future experiments.

All congurations for the hyperparameters, results file, networking were refactored out and put inside a json formatted conguration file. The programs are also setup
to use CPU or GPU to do the training through the program args. The bulk of the experiments was conducted in synthesize results.py and train.py which reduce the networking and communication time of federated learning to communication through shared memory. Model parameters are passed in memory and different objects from the Federated.py classes are created to represent each local node and the aggregation server.
![Design](https://github.com/UT-MPC/LocalGlobalFedLearning/raw/master/Papers/scratch/softwaredesign.png)

#### Hyperparameter Search
To perform the hyperparameter search, the hyperparameter_search.py program uses random search to train models using different learning rates and batch sizes. Each configuration is averaged over N runs where N is configurable. A separate validation set is create by the data loader to perform the hyperparameter optimizations. Random search is used because it is possible that one of the hyperparamaters is more important than the other so we want to be able to sample finer resolution than grid search. Each training session runs until the validation accuracy not improved for 20 epochs. The hyperparameter search provides the 20 highest validation accuracy and then the researcher  manually chooses the hyperparameter configuration and entered into the configuration file. The parameters themselves are also optimized using ADAM which is a one of the top momentum based optimizers being used at the time of this writing.

#### Deployed Version
In the deployed version of the experiments, each client runs client.py which will perform the local training and send the model weights to a central server. The model parameters themselves are saved inside a file and sent to the central server running node.js using a HTTP request. The clients periodically poll the server until all the federated averaging has been done. The server listens for all HTTP request and saves each file. Once the server has received all the HTTP request and files, it calls server.py to perform federated averaging on all the local parameters. Once the federated averaging is finished, then it sends the updated parameters to all the clients.

## Model Complexity
These models are found in `/FedMnist/model/nn_architectures.py`. 

#### Single Layer ReLu Model
This is our simplest model. Fully Connected Neural Network with one hidden layer composed of thirty neurons. Each neuron uses the ReLu activation function. A output is generated using a log softmax.

#### Four Layer Convolutional and ReLu Model
This is our second most simplest model. A convoluted neural network with two 5x5 convolutional layers. Both convolutional layers have a max pooling layer. The second convolutional layer has a drop out of 0.5. The convolutional layers are followed by two fully connected layers. The first has 320 hidden neurons and the second has 50 hidden neurons and. The first hidden layer has a drop out of 0.5. The output is generated using log softmax.

#### Six Layer Extra Wide Convolutional and ReLu Model
This is our most complex model. The convlution block has three 3x3x2 convolutional layers of stride 1 and padding 1. Each convolutional layer is followed by a batch norm layer, a ReLu layer. The second and third layers have a max pooling layer. The convolutional layers are followed by three fully connect layers with 6272, 64, and 10 hidden neurons. Each fully connected layer is followed by a batch norm layer and has a drop out of 0.5.  The output is generated using log softmax.

#### Initialization, Activation Function and Hyper Parameter Search
We used ReLu activation function and Kaiming He initialization modeled by ReLu non-linearity. At the time of the writing, squeaky ReLu has been shown to be decisively better or worse. For networks with drop out, we used a fixed 0.5 drop out. Then we performed random search for the batch size and learning rate. For each search, we train for N epochs where for epochs N+1 to N+20 there was no improvement in the validation accuracy. We constrained the batch size to be a power of two because GPU memory is configured in powers of two. 

## Usage
The federated learning library does hyperparameter search and generation of results.
#### Hyperparameter Search
Random Search for Hyperparameter
```
    python3 search_hyperparameter.py -g 0

Flag    Purpose   
------  ----------
-g      GPU Number
```
View top 20 results for Hyperparameter (use negative number)
```
    python3 search_hyperparameter.py -g -1

Flag    Purpose   
------  ----------
-g      File to pull results for.
```
#### Result Generation
```
    python3 synthesize_results.py -p 2 -g 0 -s 0 -l 3 

Flag    Purpose   
------  ----------
-p      Program Type (0 = Full Connected, Balanced)
                     (1 = Full Connected, Unbalanced)
                     (2 = Narrow ConvNet, Balanced)
                     (3 = Narrow ConvNet, Unbalanced)
                     (4 = Wide ConvNet, Balanced)
                     (5 = Wide ConvNet, Unbalanced)
-g      GPU Number
-s      Start Percentage Unbalanced in Dataloader
-l      Locality (1 = Federated Learning)
                 (2 = Local Learning)
                 (3 = Selective Federated Learning)
```
## Federated Learning Deployment
#### Server
Tested on Digital Ocean server running `Ubuntu 18.04.3 `
Inside `/FedMnist/server`
```
    node app.js
```
#### Client
Tested on Raspberry Pi 4 (4GB) and Nvidia Jetson Nano. 
For raspberry pi, install pytorch using the pytorch wheels found in `/RaspberryPi4`. To native compile a newer pytorch version, follow this [guide](https://wormtooth.com/20180617-pytorch-on-raspberrypi/). For Jetson Nano, install pytorch using pytorch wheels found [here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano-version-1-3-0-now-available/)
```
    python3 federated_client.py 0
```
## Config
#### machine_learning
```
        "N_PARTITIONS": Number of Partitions for Centralized Learning
        "EPOCH_SATURATION": For epoch search, trains until validation accuracy doesn't improve for num: EPOCH_SATURATION 
        "MAX_EPOCHS": For epoch search, hard stop for the number of EPOCHs to train for 
```
#### machine_learning - NetFC_1
```
            "FC_N_EPOCHS": Number of Epochs
            "FC_BATCH_SIZE": Batch Size
            "FC_LEARNING_RATE": Learning Rate
```
#### machine_learning - NetCNN_conv2_relu3
```
            "C2R3_N_EPOCHS": Number of Epochs 
            "C2R3_BATCH_SIZE": Batch Size
            "C2R3_LEARNING_RATE": Learning Rate
```
#### machine_learning - NetCNN_convrelu3_relu3
```
            "CR3R3_N_EPOCHS": Number of Epochs
            "CR3R3_BATCH_SIZE": Batch Size
            "CR3R3_LEARNING_RATE": Learning Rate
```
#### machine_learning - selective_aggregation
```
            "SELECTIVE_AGGREGATION_N_PARTITIONS": Number of Nodes
            "SIMILAR_COUNT": Number of ways to parition the data (eg 3 = Classes 0-3, 4-6, 7-9)
            "SIMULARITY_THRESHOLD": Hyperparameter for grouping similar nodes
```
#### hyperparameter_search
Destination files to store hyperparameter searches
```
        "FED_NETFC1_BALANCED100_FILE": Result File for Hyperparameter Search for NETFC1
        "FED_NETC2R3_BALANCED100_FILE": Result File for Hyperparameter Search for NETC2R3
        "FED_NETCR3R3_BALANCED100_FILE": Result File for Hyperparameter Search for NETCR3R3
```
#### results
Destination files to store results:
```
        "FED_NETFC1_TEST_BALANCED_FILE": Results for Federated NETFC1 Balanced Test Data
        "FED_NETFC1_TEST_UNBALANCED_FILE": Results for Federated NETFC1 Unbalanced Test Data
        "FED_NETC2R3_TEST_BALANCED_FILE": Results for Federated NETC2R3 Balanced Test Data
        "FED_NETC2R3_TEST_UNBALANCED_FILE": Results for Federated NETC2R3 Unbalanced Test Data
        "FED_NETCR3R3_TEST_BALANCED_FILE": Results for Federated NETC3R3 Balanced Test Data
        "FED_NETCR3R3_TEST_UNBALANCED_FILE": Results for Federated NETC3R3 Unbalanced Test Data
        "LOCAL_NETFC1_TEST_BALANCED_FILE": Results for Local NETFC1 Balanced Test Data
        "LOCAL_NETFC1_TEST_UNBALANCED_FILE": Results for Local NETFC1 Unbalanced Test Data
        "LOCAL_NETC2R3_TEST_BALANCED_FILE": Results for Local NETC2R3 Balanced Test Data
        "LOCAL_NETC2R3_TEST_UNBALANCED_FILE": Results for Local NETC2R3 Unbalanced Test Data
        "LOCAL_NETCR3R3_TEST_BALANCED_FILE": Results for Local NETC3R3 Balanced Test Data
        "LOCAL_NETCR3R3_TEST_UNBALANCED_FILE": Results for Local NETC3R3 Unbalanced Test Data
        "CENTRAL_NETFC1_TEST_BALANCED_FILE": Results for NETC3R3 Centralized Data
        "CENTRAL_NETC2R3_TEST_BALANCED_FILE": Results for NETC3R3 Centralized Data
        "CENTRAL_NETCR3R3_TEST_BALANCED_FILE": Results for NETC3R3 Centralized Data
        "SELECTIVE_FED_NETFC1_TEST_BALANCED_FILE":  Results for Selective Federated NETFC1 Balanced Data
        "SELECTIVE_FED_NETC2R3_TEST_BALANCED_FILE": Results for Selective Federated  NETC2R3 Balanced Data
        "SELECTIVE_FED_NETCR3R3_TEST_BALANCED_FILE": Results for Selective Federated  NETC3R3 Balanced Data
        "SELECTIVE_FED_NETFC1_TEST_UNBALANCED_FILE": Results for Selective Federated  NETFC1 Unbalanced Data
        "SELECTIVE_FED_NETC2R3_TEST_UNBALANCED_FILE": Results for Selective Federated  NETC2R3 Unbalanced Data
        "SELECTIVE_FED_NETCR3R3_TEST_UNBALANCED_FILE": Results for Selective Federated  NETFC1 Unbalanced Data
```
#### networking
```
        "SERVER_URL_BASE": IP and Port number of server
        "UPLOAD_ROUTE": Route to Upload Parameters from Local Updates
        "DOWNLOAD_ROUTE": Route to Download Parameters from Server
        "FED_AVG_DONE_ROUTE": Route to signal that Federated Averaging is done
        "QUERY_GLOBAL_SYNC_DONE_ROUTE": Route to query if all local nodes have downloaded parameters
        "QUERY_FED_AVG_DONE_ROUTE": Route to query if Federated Averaging is done
```
#### server_only
```
        "N_partitions": Number of paritions of data which in current version of code is same as number of nodes.
```
## Branches

| Branch Name     | Purpose    | 
| ------------- |:-------------:|
| master     | master | 
| Deployment     | Deployment on Physical Devices | 
| More_nodes     | Adding Support for More Nodes | 
| Production     | Generates Federated Learning Research Results | 


## Papers
- [Federated Learning Model Complexity vs Robustness to non-IID data and Selective Federated Learning](https://github.com/UT-MPC/LocalGlobalFedLearning/blob/master/Papers/ConfPaper.pdf)
