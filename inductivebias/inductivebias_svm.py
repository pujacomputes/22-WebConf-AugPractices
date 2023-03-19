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
from evaluate_embeddings import *


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

                # data = data[0]
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
        # return F.log_softmax(x, dim=-1)
        return x

    def forward_rand(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(self.args.device)
        with torch.no_grad():
            self.encoder.eval()
            x, _ = self.encoder(x, edge_index, batch)
        x = self.fc1(x)
        return x
        # return F.log_softmax(x, dim=-1)

    def forward_bn(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(self.args.device)
        with torch.no_grad():
            self.encoder.eval()
            x, _ = self.encoder(x, edge_index, batch)
        x = self.bn_int(x)
        x = self.fc1(x)
        return x
        # return F.log_softmax(x, dim=-1)


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
        # loss = F.nll_loss(output, data.y)
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
    parser.add_argument("--num_layers", type=int, default=5)
    args = parser.parse_args()
    return args


def main():

    args = arg_parser()
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", args.dataset)

    print("=> MODE: ", args.mode)
    print("=> DATASET: ", args.dataset)

    dataset = TUDataset(
        root="/home/sc/eslubana/graphssl/PosGraphCL/unsupervised_TU/data/{}".format(
            args.dataset
        ),
        name=args.dataset,
    )  # .shuffle()
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    num_graphs = len(dataset)
    print("Number of graphs", len(dataset))
    dataset = dataset.shuffle()
    args.num_classes = dataset.num_classes
    print("Num Classes", args.num_classes)
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    args.num_features = dataset_num_features
    print("Num Features: ", args.num_features)

    device = "cuda:0"
    model = Encoder(args.num_features, 32, num_gc_layers=args.num_layers).to(device)
    args.device = device

    model.eval()
    emb, y = model.get_embeddings(dataloader)
    val_acc, val_acc_std, acc, acc_std = linearsvc_classify(emb, y, search=True)
    log_str = "{dataset}\t{num_layers}\t{val_acc:.4f}\t{val_acc_std:.4f}\t{acc:.4f}\t{acc_std:.4f}".format(
        dataset=args.dataset,
        num_layers=args.num_layers,
        val_acc=val_acc,
        val_acc_std=val_acc_std,
        acc=acc,
        acc_std=acc_std,
    )
    print(log_str)
    with open("inductive_bias_svc.log", "a+") as f:
        f.write(log_str)
        f.write("\n")


if __name__ == "__main__":
    main()
