import federated

import torch

def accuracy(model, data_loader):
    correct = 0
    model.eval()
    with torch.no_grad():
        for test_x, test_y in data_loader:
            test_x = test_x.to(federated.get_device())
            test_y = test_y.to(federated.get_device())
            output = model(test_x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(test_y.view_as(pred)).sum().item()
        acc = (100 * correct) / len(data_loader.dataset)

    return acc

def selective_aggregating_vector(model, data_loader):
    correct = 0
    model.eval()
    i = 0
    with torch.no_grad():
        for test_x, test_y in data_loader:
            i = i + 1
            test_x = test_x.to(federated.get_device())
            test_y = test_y.to(federated.get_device())
            output = model(test_x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(test_y.view_as(pred)).sum().item()
        acc = (100 * correct) / len(data_loader.dataset)

    return pred.eq(test_y.view_as(pred))

def loss(model, data_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        for test_x, test_y in data_loader:
            test_x = test_x.to(federated.get_device())
            test_y = test_y.to(federated.get_device())
            outputs = model(test_x)
            loss = loss_fn(outputs, test_y)

    return loss 