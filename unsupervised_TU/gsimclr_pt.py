import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# from core.encoders import *

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb
import collections
from sklearn.metrics.pairwise import cosine_similarity


class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1.0, gamma=0.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = "fd"
        measure = "JSD"
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = -(term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, args, alpha=0.5, beta=1.0, gamma=0.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(args.dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        y = self.proj_head(y)

        return y

    def loss_cal(self, x, x_aug):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", x, x_aug) / torch.einsum(
            "i,j->ij", x_abs, x_aug_abs
        )
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()

        return loss


import random


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":

    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {"val": [], "test": []}
    epochs = 50
    log_interval = 1
    batch_size = 128
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), ".", "data", DS)

    dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug="none").shuffle()

    try:
        args.dataset_num_features = dataset.get_num_feature()
    except:
        args.dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = simclr(args.hidden_dim, args.num_gc_layers, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("================")
    print("dataset len: ", len(dataset))
    print("lr: {}".format(lr))
    print("num_features: {}".format(args.dataset_num_features))
    print("hidden_dim: {}".format(args.hidden_dim))
    print("num_gc_layers: {}".format(args.num_gc_layers))
    print("================")

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)
    pos_num = len(np.where(y == 0)[0])
    neg_num = len(np.where(y != 0)[0])

    print("Positive samples: ", pos_num)
    print("Negative samples: ", neg_num)

    pos_idx = np.random.permutation(pos_num)
    pos_idx_1 = np.random.permutation(pos_idx)

    neg_idx = np.random.permutation(neg_num)
    neg_idx_1 = np.random.permutation(neg_idx)

    emb, y = model.encoder.get_embeddings(dataloader_eval)
    losses = collections.defaultdict(list)

    for epoch in range(1, epochs + 1):
        loss_all = 0
        model.train()
        for data in dataloader:

            data, data_aug = data
            optimizer.zero_grad()

            node_num, _ = data.x.size()
            data = data.to(device)
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)

            if (
                args.aug == "dnodes"
                or args.aug == "subgraph"
                or args.aug == "random2"
                or args.aug == "random3"
                or args.aug == "random4"
            ):
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [
                    n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])
                ]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                edge_idx = [
                    [idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]]
                    for n in range(edge_num)
                    if not edge_idx[0, n] == edge_idx[1, n]
                ]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)
            x_aug = model(
                data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs
            )

            loss = model.loss_cal(x, x_aug)
            loss_all += loss.item() / data.num_graphs
            loss.backward()
            optimizer.step()

        # get the similarity scores
        model.eval()
        emb, y = model.encoder.get_embeddings(dataloader_eval)

        # pos vs pos
        p1 = emb[pos_idx]
        p2 = emb[pos_idx_1]
        n1 = emb[neg_idx]
        n2 = emb[neg_idx_1]
        pos_v_pos = (
            torch.nn.functional.cosine_similarity(
                torch.tensor(p1), torch.tensor(p2), dim=1
            )
            .mean()
            .item()
        )

        neg_v_neg = (
            torch.nn.functional.cosine_similarity(
                torch.tensor(n1), torch.tensor(n2), dim=1
            )
            .mean()
            .item()
        )

        # pos_v_pos = cosine_similarity(p1, p2).mean()
        # neg_v_neg = cosine_similarity(n1, n2).mean()

        num_pairs = min(pos_num, neg_num)
        p1 = p1[:num_pairs]
        n2 = n2[:num_pairs]
        # pos_v_neg = cosine_similarity(p1, n2).mean()

        pos_v_neg = (
            torch.nn.functional.cosine_similarity(
                torch.tensor(p1), torch.tensor(n2), dim=1
            )
            .mean()
            .item()
        )

        emb, y = model.encoder.get_embeddings(dataloader_eval)
        std_mean = (
            (emb / np.linalg.norm(emb, ord=2, axis=1, keepdims=True)).std(axis=0).mean()
        )

        losses["pvp"].append(pos_v_pos)
        losses["nvn"].append(neg_v_neg)
        losses["pvn"].append(pos_v_neg)
        losses["std"].append(std_mean)
        losses["epoch_loss"] = loss_all / len(dataloader)
        print(
            "Epoch {epoch}, Loss {loss:.4f} PvP: {pvp:.4f} NvN:{nvn:.4f} PvN:{pvn:.4f} Std:{std:.4f}".format(
                epoch=epoch,
                loss=loss_all / len(dataloader),
                pvp=pos_v_pos,
                nvn=neg_v_neg,
                pvn=pos_v_neg,
                std=std_mean,
            )
        )
        if epoch % log_interval == 0:
            model.eval()
            try:
                acc_val, acc = evaluate_embedding(emb, y)
                accuracies["val"].append(acc_val)
                accuracies["test"].append(acc)
                print(
                    "Val Acc: {val_acc:.4f} Test Acc: {test_acc:.4f} Std: {std:.4f}".format(
                        val_acc=accuracies["val"][-1],
                        test_acc=accuracies["test"][-1],
                        std=std_mean,
                    )
                )
            except:
                accuracies["val"].append(np.nan)
                accuracies["test"].append(np.nan)
                print("Skipping eval!")
                print()

    tpe = ("local" if args.local else "") + ("prior" if args.prior else "")
    with open("logs/log_" + args.DS + "_" + args.aug, "a+") as f:
        s = json.dumps(accuracies)
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s
            )
        )
        f.write("\n")

    ckpt = {}
    ckpt["net"] = model.state_dict()
    ckpt["stats"] = losses
    ckpt["acc"] = accuracies
    torch.save(ckpt, "{}_{}.pkl".format(args.DS, args.seed))
