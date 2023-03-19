import torch
import torch.backends.cudnn as cudnn
import torch_geometric as geom
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.manual_seed(0)
np.random.seed(0)
cudnn.deterministic = True
cudnn.benchmark = False

import collections
from copy import deepcopy
from utils import dotdict
import tqdm
from models import MNIST_GNN, SimSiam, SimCLR, BYOL
from datasets_mnist import *
from knn_monitor import knn_monitor
import pdb
import sys
import argparse
from torch.utils.data import Subset 
def test(model, loader, device,is_aug=False):

    correct = 0
    total = 0
    for data in loader:
        if is_aug:
            x = data[0][0].to(device)
            y = data[1].to(device)
        else: 
            x = data[0].to(device)
            y = data[1].to(device)
        with torch.no_grad():
            _,pred = model(x.x,x.edge_index,x.batch)
            pred = pred.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = {}
    args["input_dim"] = 3
    args["hidden_dim"] = 110
    args["out_dim"] = 110
    args["num_layers"] = 4
    args["readout"] = "mean"
    args["num_mlp_layers"] = 2
    args["num_classes"] = 10
    args["epochs"] = 80
    #args["lr"] = 0.03
    args["seed"] = 41
    args["readout"] = "mean"
    args["project_dim"] = 1028
    args["bottle_neck_dim"] = 128
    args["predictor_dim"] = 1028
    args["momentum"] = 0.9
    args["weight_decay"] = 1e-4
    args["device"] = device
    args["label_dim"] = 10
    args["t_max"] = 8
    args = dotdict(args)
    args.dataset = sys.argv[1]
    args.aug_type = sys.argv[2]
    args.aug_ratio = float(sys.argv[3])
    args.optim = sys.argv[4]
    args.lr = float(sys.argv[5])
    args.batch_size = int(sys.argv[6])
    args.seed = int(sys.argv[7])
    args.load_ckpt = sys.argv[8]
    args.supervised=False
    print("=" * 50)
    print("USING DATASET: ", args.dataset)
    print("USING AUGMENTATION: ", args.aug_type)
    print("USING AUG RATIO: ", args.aug_ratio)
    print("USING OPTIM: ", args.optim)
    print("USING OPTIM: ", args.batch_size)
    print("USING LEARNING RATE: ", args.lr)
    print("USING BATCH SIZE: ", args.batch_size)
    print("USING SEED: ", args.seed)
    print("SUPERVISED?: ", args.supervised)
    print("Device: ",device)
    print("Load Ckpt: ",args.load_ckpt)
    print("=" * 50)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root_dir = "data/"
    args.input_dim = 5
    if args.aug_type == "GRAPH":
        train_augmentation = MNISTGraphTransform(aug_ratio=args.aug_ratio)
        eval_augmentation = MNISTEvalTransform()
        print("=> SELECTED MNIST GRAPH")
    elif args.aug_type == "IMAGE":
        train_augmentation = MNISTImageTransform()
        eval_augmentation = MNISTEvalTransform()
        print("=> SELECTED MNIST IMAGE")
    elif args.aug_type == "BOTH":
        train_augmentation = MNISTBothTransform(aug_ratio=args.aug_ratio)
        eval_augmentation = MNISTColorEvalTransform()
        print("=> SELECTED BOTH")
    elif args.aug_type == "COLOR":
        train_augmentation = MNISTColorizeTransform()
        eval_augmentation = MNISTColorEvalTransform()
        print("=> SELECTED MNIST COLOR")
    else:
        print("ERROR AUG NOT FOUND")

    clean_test_dataset = torchvision.datasets.MNIST(
        root=root_dir, download=True, train=False, transform=eval_augmentation
    )    
    print("Train Samples: ", len(clean_test_dataset))
    
    """
    INITIALIZE AND TRAIN MODEL!
    """
    backbone = MNIST_GNN(args).to(device)
    model = BYOL(backbone=backbone, args=args).to(device)
    ckpt = torch.load(args['load_ckpt'])
    model.load_state_dict(ckpt['net'])
    model.eval() 
    #extra all representations => sort them by class. 
    feature_bank = []
    for class_enum in range(10):
        labels = np.random.choice((clean_test_dataset.test_labels == class_enum).nonzero().reshape(-1),size=100,replace=False)
        t_l = DataLoader(Subset(clean_test_dataset,labels), 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4)
        with torch.no_grad():
            for data in t_l:
                data = data[0].to(args.device)
                feature = model.encoder[0](data.x, data.edge_index,data.batch)
                feature = torch.nn.functional.normalize(feature, dim=1)
                feature_bank.append(feature.cpu())
        print("Done: {}/10".format(class_enum))
        
    feature_bank = torch.cat(feature_bank, dim=0)
    cos_sim = feature_bank @ feature_bank.t()
    print('cos_sim',cos_sim.shape) 
    save_str = "SIM_MATRICES/byol_{aug_type}_{aug_ratio}_{seed}.ckpt".format(aug_type=args.aug_type,aug_ratio=args.aug_ratio,seed=args.seed)
    torch.save(cos_sim,save_str)
if __name__ == "__main__":
    main()
