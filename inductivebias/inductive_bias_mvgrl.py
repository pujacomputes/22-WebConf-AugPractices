import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch_geometric
import numpy as np
from sklearn.metrics import accuracy_score
import sys
import argparse
import pdb
import torch.nn as nn
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold


class MVRL_GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(MVRL_GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = self.fc(feat)
        out = torch.bmm(adj, feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class MVRL_GCN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=32, num_layers=5, device="cuda", args=None
    ):
        super(MVRL_GCN, self).__init__()
        n_h = out_ft = hidden_dim
        in_ft = input_dim
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.layers = []
        self.num_layers = num_layers
        self.conv1 = MVRL_GCNLayer(in_ft, n_h).cuda()
        self.conv2 = MVRL_GCNLayer(n_h, n_h).cuda()
        self.conv3 = MVRL_GCNLayer(n_h, n_h).cuda()
        self.conv4 = MVRL_GCNLayer(n_h, n_h).cuda()

        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, data):
        batch_size = data.num_graphs
        xs = []
        x, _ = torch_geometric.utils.to_dense_batch(
            data.x, data.batch, max_num_nodes=self.args.num_nodes
        )
        adj = torch_geometric.utils.to_dense_adj(
            data.edge_index, data.batch, max_num_nodes=self.args.num_nodes
        )
        x = self.layers[0](x, adj)
        xs.append(x)
        for idx in range(3):
            x = self.layers[idx + 1](x, adj)
            xs.append(x)

        xs = [x.sum(dim=1).squeeze(1) for x in xs]
        x = torch.cat(xs, 1)
        return x


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
        self.encoder = MVRL_GCN(
            input_dim=args.num_features,
            hidden_dim=32,
            num_layers=5,
            device="cuda",
            args=args,
        )
        self.args = args
        self.fc1 = Linear(dim * 4, args.num_classes)

    def forward_e2e(self, data):
        if data.x is None:
            x = torch.ones(data.num_graphs).to(self.args.device)
        x = self.encoder(data)
        x = self.fc1(x)
        #return F.log_softmax(x, dim=-1)
        return x 

    def forward_rand(self, data):
        if data.x is None:
            data.x = torch.ones(data.num_graphs).to(self.args.device)
        with torch.no_grad():
            self.encoder.eval()
            x = self.encoder(data)
        x = self.fc1(x)
        #return F.log_softmax(x, dim=-1)
        return x 

    def forward_bn(self, data):
        if data.x is None:
            data.x = torch.ones(data.num_graphs).to(self.args.device)
        with torch.no_grad():
            self.encoder.eval()
            x = self.encoder(data)
        x = self.fc1(x)
        #return F.log_softmax(x, dim=-1)
        return x 


def train(model, train_loader, optimizer, epoch, args):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(args.device)
        optimizer.zero_grad()
        if args.mode == "e2e":
            output = model.forward_e2e(data)
        elif args.mode == "rand":
            output = model.forward_rand(data)
        elif args.mode == "bn":
            output = model.forward_bn(data)
        else:
            print("ERROR!")
        loss = F.nll_loss(output, data.y)
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
            output = model.forward_e2e(data)
        elif args.mode == "rand":
            output = model.forward_rand(data)
        elif args.mode == "bn":
            output = model.forward_bn(data)
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
    parser.add_argument("--num_nodes", type=int, default=50)
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

    dataset = TUDataset(
        path,
        name=args.dataset,
    )  # .shuffle()
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
    with open("inductive_bias_mvgrl.log", "a+") as f:
        log_str = "{dataset}\t{mode}\t{epoch:.4f}\t{max_acc:.4f}\t{std:.4f}\t{f_acc:.4f}\t{f_std:.4f}\{batch_size}".format(
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
