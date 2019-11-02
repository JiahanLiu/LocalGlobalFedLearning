import torch

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

def evaluate(model, test_loader):
    acc = -1
    correct = 0
    model.eval()
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(DEVICE)
            test_y = test_y.to(DEVICE)
            output = model(test_x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(test_y.view_as(pred)).sum().item()
        acc = (100 * correct) / len(test_loader.dataset)

    return acc