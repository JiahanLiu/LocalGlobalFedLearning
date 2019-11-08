import torch

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

def accuracy(model, data_loader):
    correct = 0
    model.eval()
    with torch.no_grad():
        for test_x, test_y in data_loader:
            test_x = test_x.to(DEVICE)
            test_y = test_y.to(DEVICE)
            output = model(test_x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(test_y.view_as(pred)).sum().item()
        acc = (100 * correct) / len(data_loader.dataset)

    return acc

def loss(model, data_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        for test_x, test_y in data_loader:
            test_x = test_x.to(DEVICE)
            test_y = test_y.to(DEVICE)
            outputs = model(test_x)
            loss = loss_fn(outputs, test_y)

    return loss