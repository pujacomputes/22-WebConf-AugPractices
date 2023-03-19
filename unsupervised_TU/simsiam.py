import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from math import pi, cos
import collections

# from core.encoders import *

from aug import TUDataset_aug as TUDataset

# from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
import pdb
from arguments import arg_parse, get_model_id
import copy


class multiSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


BASE_TARGET_EMA = 4e-3


class simsiam(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, bottleneck_dim):
        # def __init__(self, backbone):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_gc_layers = num_gc_layers
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.backbone = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        # self.projector = MLP(backbone.output_dim)

        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.encoder = multiSequential(self.backbone, self.projector)

        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, self.embedding_dim),
        )

    def D(self, p, z, version="simplified"):  # negative cosine similarity
        if version == "original":
            z = z.detach()  # stop gradient
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif (
            version == "simplified"
        ):  # same thing, much faster. Scroll down, speed test in __main__
            return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            raise Exception

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1 = f[1](f[0](x1.x, x1.edge_index, x1.batch)[0])
        z2 = f[1](f[0](x2.x, x2.edge_index, x2.batch)[0])
        p1, p2 = h(z1), h(z2)

        return z1, z2, p1, p2

    def loss(self, z1, z2, p1, p2):
        L = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        return L


if __name__ == "__main__":

    args = arg_parse()
    accuracies = accuracies = {"val": [], "test": []}
    epochs = args.epochs
    log_interval = 1
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS

    model_id = get_model_id("simsiam",args)
    print("MODEL_ID: ",model_id)

    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", DS)
    dataset = TUDataset(path, name=DS, aug=args.aug, aug_ratio=args.aug_ratio).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug="none").shuffle()
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = simsiam(args.hidden_dim, args.num_gc_layers, args.bottleneck_dim).to(device)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_epoch = 0.0
    best_ckpt = {}
    losses = collections.defaultdict(list)
    val_acc = np.nan
    print("================")
    print("Dataset Size: ",len(dataset))
    print("lr: {}".format(lr))
    print("num_features: {}".format(dataset_num_features))
    print("hidden_dim: {}".format(args.hidden_dim))
    print("Epochs: {}".format(args.epochs))
    print("num_gc_layers: {}".format(args.num_gc_layers))
    print("================")

    args.skip_counter = 0
    for epoch in range(1, epochs + 1):
        loss_all = 0
        model.train()

        running_loss = 0.0
        running_backbone_norm = 0.0
        running_encoder_norm = 0.0
        running_predictor_norm = 0.0
        running_backbone_std = 0.0
        running_encoder_std = 0.0
        running_predictor_std = 0.0 

        for data in dataloader:

            # print('start')
            data, data_aug = data
            optimizer.zero_grad()

            node_num, _ = data.x.size()
            data = data.to(device)

            if (
                args.aug == "dnodes"
                or args.aug == "subgraph"
                or args.aug == "random2"
                or args.aug == "random3"
                or args.aug == "random4"
            ):
                # node_num_aug, _ = data_aug.x.size()
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

            try:
                z1, z2, p1, p2 = model(data, data_aug)
                loss = model.loss(z1, z2, p1, p2)
                #print(loss.item())

                loss_all += loss.item()
                loss.backward()
                optimizer.step()

                #iteration-level bookkeeping
                model.eval()
                with torch.no_grad():
                    running_loss += loss.item()
                    
                    b1= model.encoder[0](data.x,
                                            data.edge_index,
                                            data.batch)[0]
                    
                    b2 = model.encoder[0](data_aug.x,
                                            data_aug.edge_index, 
                                            data_aug.batch)[0]
                    
                    p1 = model.encoder[1](b1)
                    
                    p2 = model.encoder[1](b2)
                    
                    z1 = model.predictor(p1)
                    z2 = model.predictor(p2)
                    
                    #similarities
                    b_sim = torch.nn.functional.cosine_similarity(b1,b2,dim=1).mean()
                    p_sim = torch.nn.functional.cosine_similarity(p1,p2,dim=1).mean()
                    z_sim = torch.nn.functional.cosine_similarity(z1,z2,dim=1).mean()
                    
                    #norms & std
                    running_backbone_norm += b1.norm(dim=1).mean()
                    running_backbone_std += (b1 / (b1.norm(dim=1,keepdim=True) + 1e-10)).std(dim=0).mean()
                    
                    running_encoder_norm += p1.norm(dim=1).mean()
                    running_encoder_std += (p1 / (p1.norm(dim=1,keepdim=True) + 1e-10)).std(dim=0).mean()
                    
                    running_predictor_norm += z1.norm(dim=1).mean()
                    running_predictor_std += (z1 / (z1.norm(dim=1,keepdim=True)+ 1e-10)).std(dim=0).mean()

                    losses['backbone_sim'].append(b_sim.item())
                    losses['encoder_sim'].append(p_sim.item())
                    losses['predictor_sim'].append(z_sim.item())
                model.train()
            except:
                args.skip_counter += 1
                print("*"*50)
                print(" SKIPPING SAMPLE: {}".format(skip_count))
                print("*"*50)
        
        print()
        print("Epoch {}, Loss {}".format(epoch, loss_all / len(dataloader)))
        print()

        model.eval()     
        if epoch % log_interval == 0:
            emb, y = model.backbone.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies["val"].append(acc_val)
            accuracies["test"].append(acc)
            # print(accuracies['val'][-1], accuracies['test'][-1])
 
        running_loss /= dataloader.__len__()
        running_backbone_norm /=  dataloader.__len__()
        running_backbone_std /=  dataloader.__len__()
        running_encoder_norm /=  dataloader.__len__()
        running_encoder_std /=  dataloader.__len__()
        running_predictor_norm /=  dataloader.__len__()
        running_predictor_std /=  dataloader.__len__()

        losses['epoch_loss'].append(running_loss)
        losses["val_acc"].append(acc_val)
        losses["acc"].append(acc)
        losses['backbone_norm'].append(running_backbone_norm)
        losses['backbone_std'].append(running_backbone_std.item())
        losses['encoder_norm'].append(running_encoder_norm)
        losses['encoder_std'].append(running_encoder_std.item())
        losses['predictor_norm'].append(running_predictor_norm)
        losses['predictor_std'].append(running_predictor_std.item())

        if acc_val > best_epoch:
            best_ckpt["unsupervised_best_epoch"] = epoch
            best_ckpt["val_acc"] = acc_val
            best_ckpt['model'] = model.state_dict()
            best_ckpt["args"] = vars(args)
            best_ckpt['stats'] = losses
            best_epoch = acc_val
            torch.save(best_ckpt, "CKPTS/best_{}.pth".format(model_id))
            print("Epoch {0} -- Best Acc: {1:.4f} ".format(epoch, best_epoch))
    ## make ckpt and save! 
    final_ckpt = {}
    final_ckpt["unsupervised_best_epoch"] = epoch
    final_ckpt["val_acc"] = acc_val
    final_ckpt['model'] = model.state_dict()
    final_ckpt["args"] = vars(args)
    final_ckpt['stats'] = losses
    torch.save(final_ckpt, "CKPTS/final_{}.pth".format(model_id))

    print("TOTAL SKIPS: ",args.skip_counter)

    tpe = ("local" if args.local else "") + ("prior" if args.prior else "")
    with open("logs/log_" + args.DS + "_" + args.aug, "a+") as f:
        s = json.dumps(accuracies)
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s
            )
        )
        f.write("\n")
