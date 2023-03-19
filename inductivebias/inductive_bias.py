import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import sys
import argparse
import pdb


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
            # if i == 2:
            # feature_map = x2

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def get_embeddings_v(self, loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                    break

        return x_g, ret, y


class SimpleNet(torch.nn.Module):
    def __init__(self, args):
        super(SimpleNet, self).__init__()
        try:
            num_features = args.num_features
        except:
            num_features = 1
        dim = 32

        if args.mode == "bn":
            self.bn_int = torch.nn.BatchNorm1d(dim * 5)
        self.encoder = Encoder(num_features, dim, num_gc_layers=5)
        self.args = args
        self.fc1 = Linear(dim * 5, args.num_classes)

    def forward_e2e(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(self.args.device)
        # with torch.no_grad():
        #     self.encoder.eval()
        #     x, _ = self.encoder(x, edge_index, batch)
        x, _ = self.encoder(x, edge_index, batch)
        x = self.fc1(x)
        #return F.log_softmax(x, dim=-1)
        return x

    def forward_rand(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(self.args.device)
        with torch.no_grad():
            self.encoder.eval()
            x, _ = self.encoder(x, edge_index, batch)
        x = self.fc1(x)
        return x
        #return F.log_softmax(x, dim=-1)

    def forward_bn(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(self.args.device)
        with torch.no_grad():
            self.encoder.eval()
            x, _ = self.encoder(x, edge_index, batch)
        x = self.bn_int(x)
        x = self.fc1(x)
        return x
        #return F.log_softmax(x, dim=-1)


def train(model, train_loader, optimizer, epoch, args):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(args.device)
        optimizer.zero_grad()
        if args.mode == "e2e":
            output = model.forward_e2e(data.x, data.edge_index, data.batch)
        elif args.mode == "rand":
            output = model.forward_rand(data.x, data.edge_index, data.batch)
        elif args.mode == "bn":
            output = model.forward_bn(data.x, data.edge_index, data.batch)
        else:
            print("ERROR!")
        #loss = F.nll_loss(output, data.y)
        loss = torch.nn.CrossEntropyLoss()(output, data.y)
        loss.backward()
        loss_all += loss.item()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 2.0)
        torch.nn.utils.clip_grad_norm_(model.fc1.parameters(), 2.0)
        optimizer.step()

    return loss_all / train_loader.__len__()


def test(model, loader, args):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(args.device)
        if args.mode == "e2e":
            output = model.forward_e2e(data.x, data.edge_index, data.batch)
        elif args.mode == "rand":
            output = model.forward_rand(data.x, data.edge_index, data.batch)
        elif args.mode == "bn":
            output = model.forward_bn(data.x, data.edge_index, data.batch)
        else:
            print("ERROR!")
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Models for Measuring the Inductive Bias"
    )
    parser.add_argument("--mode", type=str, default="rand", help="e2e, bn ,rand")
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def main():

    args = arg_parser()
    epochs = 200
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", args.dataset)
    accuracies = [[] for i in range(epochs)]

    print("=> MODE: ", args.mode)
    print("=> DATASET: ", args.dataset)

    dataset = TUDataset(path, name=args.dataset)  # .shuffle()
    num_graphs = len(dataset)
    print("Number of graphs", len(dataset))
    dataset = dataset.shuffle()
    args.num_classes = dataset.num_classes
    print("Num Classes", args.num_classes)
    args.num_features = dataset.num_features
    print("Num Features: ", args.num_features)
    splits = 10
    kf = KFold(n_splits=splits, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(dataset):

        train_dataset = [dataset[int(i)] for i in list(train_index)]
        test_dataset = [dataset[int(i)] for i in list(test_index)]
        print("len(train_dataset)", len(train_dataset))
        print("len(test_dataset)", len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleNet(args).to(device)
        args.device = device
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[epochs * 0.5, epochs * 0.75], gamma=0.1
        )
        for epoch in tqdm(range(1, epochs + 1)):
            train_loss = train(model, train_loader, optimizer, epoch, args)
            train_acc = test(model, train_loader, args)
            test_acc = test(model, test_loader, args)
            accuracies[epoch - 1].append(test_acc)
            scheduler.step()
    tmp = np.mean(accuracies, axis=1)
    print(
        args.dataset,
        np.argmax(tmp),
        np.max(tmp),
        np.std(accuracies[np.argmax(tmp)]),
    )
    acc_array = np.array(accuracies).reshape(splits, epochs)
    final_acc = np.mean(np.mean(acc_array[:, -5:], axis=1))
    final_std = np.std(np.mean(acc_array[:, -5:], axis=1))
    with open("inductive_bias.log", "a+") as f:
        log_str = "{dataset}\t{mode}\t{epoch:.4f}\t{max_acc:.4f}\t{std:.4f}\t{f_acc:.4f}\t{f_std:.4f}\t{batch_size}".format(
            dataset=args.dataset,
            mode=args.mode,
            epoch=np.argmax(tmp),
            max_acc=np.max(tmp),
            std=np.std(accuracies[np.argmax(tmp)]),
            f_acc=final_acc,
            f_std=final_std,
            batch_size=args.batch_size,
        )
        f.write(log_str)
        f.write("\n")


if __name__ == "__main__":
    main()
