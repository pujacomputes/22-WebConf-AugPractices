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
from models import MNIST_GNN, SimSiam, BYOL
from datasets_mnist import *
from knn_monitor import knn_monitor
import pdb
import sys


def run_unsupervised(
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    val_loader,
    args,
    model_id,
):
    # TODO: See if Ckpt to resume
    best_epoch = 0.0
    best_ckpt = {}
    losses = collections.defaultdict(list)
    last_five = args.epochs - 5
    knn_acc = np.nan
    try:
        with tqdm.tqdm(
            total=args.epochs,
            postfix="{desc}",
            position=0,
            leave=False,
            ascii=True,
        ) as pbar:
            K = args.epochs * train_loader.__len__()
            curr_k = 0.0
            for i in range(args.epochs):
                model.train()
                running_loss = 0.0

                running_backbone_norm = 0.0
                running_encoder_norm = 0.0
                running_predictor_norm = 0.0

                running_backbone_std = 0.0
                running_encoder_std = 0.0
                running_predictor_std = 0.0

                for data in train_loader:
                    view_1 = data[0][0].to(args.device)
                    view_2 = data[0][1].to(args.device)

                    optimizer.zero_grad()
                    # compute embeddings
                    L = model(view_1, view_2)
                    # calculate loss
                    L.backward()
                    # take optimizer step
                    if args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 2.0
                        )  # CHECK THIS?
                    optimizer.step()
                    curr_k += 1
                    model.update_moving_average(global_step=curr_k,max_steps=K)
                    # bookkeeping
                    model.eval()
                    with torch.no_grad():
                        running_loss += L.item()

                        b1 = model.encoder(view_1.x, view_1.edge_index, view_1.batch)
                        b2 = model.encoder(view_2.x, view_2.edge_index, view_2.batch)

                        z1 = model.predictor(b1)
                        z2 = model.predictor(b2)

                        # similarities
                        b_sim = F.cosine_similarity(b1, b2, dim=1).mean()
                        z_sim = F.cosine_similarity(z1, z2, dim=1).mean()

                        # norms & std
                        running_backbone_norm += b1.norm(dim=1).mean()
                        running_backbone_std += (
                            (b1 / b1.norm(dim=1, keepdim=True)).std(dim=0).mean()
                        )

                        running_predictor_norm += z1.norm(dim=1).mean()
                        running_predictor_std += (
                            (z1 / z1.norm(dim=1, keepdim=True)).std(dim=0).mean()
                        )

                        losses["backbone_sim"].append(b_sim.item())
                        losses["encoder_sim"].append(b_sim.item())
                        losses["predictor_sim"].append(z_sim.item())
                    model.train()

                # KNN Acc
                knn_acc = knn_monitor(
                    net=model.encoder[0],
                    val_data_loader=val_loader,
                    test_data_loader=test_loader,
                    epoch=i,
                    args=args,
                    k=args.k,
                    t=0.1,
                    hide_progress=True,
                )

                # bookkeeping
                running_loss /= train_loader.__len__()
                running_backbone_norm /= train_loader.__len__()
                running_backbone_std /= train_loader.__len__()
                running_encoder_norm /= train_loader.__len__()
                running_encoder_std /= train_loader.__len__()
                running_predictor_norm /= train_loader.__len__()
                running_predictor_std /= train_loader.__len__()

                losses["epoch_loss"].append(running_loss)
                losses["knn_acc"].append(knn_acc)
                losses["backbone_norm"].append(running_backbone_norm)
                losses["backbone_std"].append(running_backbone_std)
                losses["encoder_norm"].append(running_encoder_norm)
                losses["encoder_std"].append(running_encoder_std)
                losses["predictor_norm"].append(running_predictor_norm)
                losses["predictor_std"].append(running_predictor_std)

                pbar.set_postfix_str(
                    "Loss: {l:.4f} -- KNN Acc: {knn:.4f} -- BBS: {bbn:.4f}  PS: {ps:.4f}".format(
                        l=losses["epoch_loss"][-1],
                        knn=knn_acc,
                        bbn=b_sim.item(),
                        ps=-1,
                    )
                )
                pbar.update(1)

                ckpt = {}
                ckpt["epoch"] = i
                ckpt["knn_acc"] = knn_acc
                ckpt["model"] = model.state_dict()
                ckpt["args"] = dict(args)
                ckpt["stats"] = losses
                torch.save(
                    ckpt,
                    "CKPTS_BYOL/{model_id}_{aug_type}_{aug_ratio}_epoch_{e}.pth".format(
                        model_id=model_id,
                        aug_type=args.aug_type,
                        aug_ratio=args.aug_ratio,
                        e=i,
                    ),
                )
                if knn_acc > best_epoch:
                    best_ckpt["unsupervised_best_epoch"] = i
                    best_ckpt["knn_acc"] = knn_acc
                    best_ckpt["model"] = model.state_dict()
                    best_ckpt["args"] = dict(args)
                    best_ckpt["stats"] = losses
                    best_epoch = knn_acc
                    torch.save(
                        best_ckpt,
                        "CKPTS_BYOL/best_{model_id}_{aug_type}_{aug_ratio}.pth".format(
                            model_id=model_id,
                            aug_type=args.aug_type,
                            aug_ratio=args.aug_ratio,
                        ),
                    )
                    print("Epoch {0} -- Best Acc: {1:.4f} ".format(i, best_epoch))

                if scheduler is not None:
                    scheduler.step()

    except KeyboardInterrupt:
        print("*" * 50)
        print("CRTL-C EARLY TRAINING INTERUPT")
        print("SAVING INTERMEDIATE CKPT")

        final_ckpt = {}
        final_ckpt["knn_acc"] = knn_acc
        final_ckpt["net"] = model.state_dict()
        final_ckpt["args"] = dict(args)
        final_ckpt["stats"] = losses
        final_ckpt["interupted_epoch"] = i
        torch.save(
            final_ckpt,
            "CKPTS_BYOL/resume_{model_id}_{aug_type}_{aug_ratio}.pth".format(
                model_id=model_id, aug_type=args.aug_type, aug_ratio=args.aug_ratio
            ),
        )
        print("*" * 50)
        return model, losses, final_ckpt
    pbar.close()
    print("*=" * 50)
    print("Mean KNN ACC: ", torch.Tensor(losses["knn_acc"][-5:]).mean())
    print("Max KNN ACC: ", torch.Tensor(losses["knn_acc"]).max())
    print("*=" * 50)

    final_ckpt = {}
    final_ckpt["knn_acc"] = knn_acc
    final_ckpt["net"] = model.state_dict()
    final_ckpt["args"] = dict(args)
    final_ckpt["stats"] = losses
    torch.save(
        final_ckpt,
        "CKPTS_BYOL/final_{model_id}_{aug_type}_{aug_ratio}.pth".format(
            model_id=model_id, aug_type=args.aug_type, aug_ratio=args.aug_ratio
        ),
    )

    return model, losses, best_ckpt


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
    print("=" * 50)
    print("USING DATASET: ", args.dataset)
    print("USING AUGMENTATION: ", args.aug_type)
    print("USING AUG RATIO: ", args.aug_ratio)
    print("USING OPTIM: ", args.optim)
    print("USING OPTIM: ", args.batch_size)
    print("USING LEARNING RATE: ", args.lr)
    print("USING BATCH SIZE: ", args.batch_size)
    print("USING SEED: ", args.seed)
    print("=" * 50)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root_dir = "data/"
    if args.dataset == "MNIST":
        args.input_dim = 5
        if args.aug_type == "GRAPH":
            train_augmentation = MNISTGraphTransform(aug_ratio=args.aug_ratio)
            eval_augmentation = MNISTEvalTransform()
            print("=> SELECTED MNIST GRAPH")
        elif args.aug_type == "IMAGE":
            train_augmentation = MNISTImageTransform()
            eval_augmentation = MNISTEvalTransform()
            print("=> SELECTED MNIST IMAGE")
        elif args.aug_type == "COLOR":
            train_augmentation = MNISTColorizeTransform()
            eval_augmentation = MNISTColorEvalTransform()
            print("=> SELECTED MNIST COLOR")
        elif args.aug_type == "BOTH":
            train_augmentation = MNISTBothTransform(aug_ratio=args.aug_ratio)
            eval_augmentation = MNISTColorEvalTransform()
        else:
            print("ERROR AUG NOT FOUND")

        dataset = torchvision.datasets.MNIST(
            root=root_dir, download=True, train=True, transform=train_augmentation
        )
        test_val_dataset = torchvision.datasets.MNIST(
            root=root_dir, download=True, train=False, transform=eval_augmentation
        )
        args.k = 10
    elif args.dataset == "CIFAR":
        args.input_dim = 3
        print("=" * 50)
        print("WARNING NOT USING POSITIONAL INFO!")
        print("=" * 50)
        if args.aug_type == "GRAPH":
            train_augmentation = CIFARGraphTransform(aug_ratio=args.aug_ratio)
            eval_augmentation = CIFAREvalTransform()
        elif args.aug_type == "IMAGE":
            train_augmentation = CIFARImageTransform()
            eval_augmentation = CIFAREvalTransform()
        elif args.aug_type == "BOTH":
            train_augmentation = MNISTBothTransform(aug_ratio=args.aug_ratio)
            eval_augmentation = MNISTColorEvalTransform()
            print("=> SELECTED BOTH")
        else:
            print("ERROR AUG NOT FOUND")
        args.k = 100

        dataset = torchvision.datasets.CIFAR10(
            root=root_dir, download=True, train=True, transform=train_augmentation
        )
        test_val_dataset = torchvision.datasets.CIFAR10(
            root=root_dir, download=True, train=False, transform=eval_augmentation
        )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
    print("Train Samples: ", len(dataset))

    print("Val+Test Samples: ", len(test_val_dataset))
    labels = []
    for d, t in test_val_dataset:
        labels.append(t)

    test_val_split = int(np.floor(len(test_val_dataset) * 0.5))
    val_loader = DataLoader(
        torch.utils.data.Subset(
            test_val_dataset, np.arange(test_val_split, len(test_val_dataset))
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    val_loader.dataset.label_all = labels[test_val_split:]

    test_loader = DataLoader(
        torch.utils.data.Subset(test_val_dataset, np.arange(0, test_val_split)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader.dataset.label_all = labels[0:test_val_split]

    """
    INITIALIZE AND TRAIN MODEL!
    """
    backbone = MNIST_GNN(args).to(device)
    model = BYOL(backbone=backbone, args=args).to(device)

    if args.optim == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        print("=> SELECTED OPTIMIZER: SGD")
    elif args.optim == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print("=> SELECTED OPTIMIZER: ADAM")
    scheduler = None
    model, loss, ckpt = run_unsupervised(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        args=args,
        model_id="{0}_{1}_bs_{2}_rep_{3}".format(args.dataset, args.optim,args.batch_size,args.seed),
    )


if __name__ == "__main__":
    main()
