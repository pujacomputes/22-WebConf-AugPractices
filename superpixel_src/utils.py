import sys
from copy import deepcopy

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __deepcopy__(self, memo=None):
        return dotdict(deepcopy(dict(self), memo=memo))


def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()


def logger(info):
    fold, epoch = info["fold"], info["epoch"]
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc = info["train_acc"], info["test_acc"]
        print(
            "{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}".format(
                fold, epoch, train_acc, test_acc
            )
        )
    sys.stdout.flush()


def test(net, data_loader, device):
    net.eval()
    correct = 0
    for data, target in data_loader:
        data.to(device)
        pred = net(data.x, data.edge_index, data.batch)
        if len(pred) > 1:
            pred = pred[1]
        pred = pred.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    acc = correct / data_loader.dataset.__len__()
    return acc
