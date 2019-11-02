import torch as torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class NetFC(nn.Module):

    def __init__(self):
        super(NetFC, self).__init__()

        self.z1 = nn.Linear(784, 30)
        self.z2_output = nn.Linear(30, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.z1(x))
        x = self.z2_output(x)
        x = F.log_softmax(x, dim=1)

        return x

# Takes too long to train on CPU, haven't validated for hyper parameters
class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
