import time
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from math import ceil

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import pdb
import os

from utils import (
    load_file,
    preprocessing,
    get_vocab,
    load_embeddings,
    create_gows,
    accuracy,
    generate_batches,
    AverageMeter,
    train,
    train_simsiam,
    load_R8,
    dotdict,
    arg_parser,
    get_splits,
    normalize,
    knn,
)
from models_v2 import MPAD, BYOL
from scipy import sparse
import collections
from aug import rand_node_augment, rand_edge_augment, rand_subgraph_augment

aug_dict = {
    "n": rand_node_augment,
    "e": rand_edge_augment,
    "s": rand_subgraph_augment,
}


def make_model_id(args):
    if args.use_nlp_aug == False:
        aug_params = []
        for aug in [args.rand_aug_1, args.rand_aug_2]:
            if aug == "n":
                aug_params.append(str(args.rand_node_drop))
            elif aug == "e":
                aug_params.append(str(args.rand_edge_perturb))
            elif aug == "s":
                aug_params.append(str(args.rand_subgraph_drop))
            else:
                print("ERROR GRAPH AUG NOT RECOGNIZED!!")
        model_path = "BYOL_GRAPH_CKPTS/"
        model_id = "_".join(
            [
                args.dataset,
                args.mp_type,
                args.optim,
                str(args.hidden),
                str(args.bottleneck),
                str(args.batch_size),
                args.rand_aug_1,
                aug_params[0],
                args.rand_aug_2,
                aug_params[1],
                "window",
                str(args.window_size),
                "rep",
                str(args.seed),
            ]
        )
        model_path += model_id

    elif args.use_nlp_aug:
        model_path = "BYOL_NLP_CKPTS/"
        model_id = "_".join(
            [
                args.dataset,
                args.mp_type,
                args.optim,
                str(args.hidden),
                str(args.bottleneck),
                str(args.batch_size),
                "-",
                "-",
                "-",
                "-",
                "window",
                str(args.window_size),
                "rep",
                str(args.seed),
            ]
        )
        model_path += model_id
    print("=> MODEL PATH: ", model_path)
    print("=> MODEL ID: ", model_id)
    try:
        os.makedirs(model_path)
    except:
        print("Could not make directory: ", model_path)
    return model_path, model_id


def main():

    args = arg_parser()
    print("***************************")
    print("MP TYPE: ", args.mp_type)
    print("***************************")
    model_path, model_id = make_model_id(args)
    # build vocabulary

    if args.use_nlp_aug:
        # need to use consolidate file!
        docs, class_labels = load_file(args.consolidated_file)
        print("\t => USING CONSOLIDATED FILE!")
    else:
        docs, class_labels = load_file(args.path_to_dataset)
    docs = preprocessing(docs)

    l_enc = LabelEncoder()
    class_labels = l_enc.fit_transform(class_labels)

    nclass = np.unique(class_labels).size
    y = list()
    for i in range(len(class_labels)):
        t = np.zeros(1)
        t[0] = class_labels[i]
        y.append(t)
    print("=> Number of Classes: ", nclass)
    args.nclass = nclass

    vocab = get_vocab(docs)
    embeddings = load_embeddings("../GoogleNews-vectors-negative300.bin", vocab)

    # set-up for nlp aug
    def load_epoch_txt(epoch_num, l_enc, args):
        file_name = "../datasets/{dataset}/{unique}/{epoch_num}.txt".format(
            dataset=args.dataset,
            unique="unique" if args.use_unique else "non_unique",
            epoch_num=epoch_num,
        )
        docs, class_labels = load_file(file_name)
        docs = preprocessing(docs)
        class_labels = l_enc.fit_transform(class_labels)
        nclass = np.unique(class_labels).size
        y = list()
        for i in range(len(class_labels)):
            t = np.zeros(1)
            t[0] = class_labels[i]
            y.append(t)
        args.nclass = nclass
        adj, features, _ = create_gows(
            docs,
            vocab,
            args.window_size,
            args.directed,
            args.normalize,
            args.use_master_node,
        )
        return adj, features, y

    adj, features, y = load_epoch_txt(0, l_enc, args)
    args.nclass = nclass = np.unique(class_labels).size
    print("=> Number of Classes: ", args.nclass)
    print("=> \t Size of Y: ", len(y))

    # create train test splits
    train_index, val_index, test_index = get_splits(y)
    n_train_batches = ceil(len(train_index) / args.batch_size)
    n_val_batches = ceil(len(val_index) / args.batch_size)
    n_test_batches = ceil(len(test_index) / args.batch_size)
    print("=> NUM TRAIN BATCHES: ", n_train_batches)
    print("=> NUM VAL BATCHES: ", n_val_batches)
    print("=> NUM TEST BATCHES: ", n_test_batches)

    y_train = [y[i] for i in train_index]
    adj_val = [adj[i] for i in val_index]
    features_val = [features[i] for i in val_index]
    y_val = [y[i] for i in val_index]
    adj_val, features_val, batch_n_graphs_val, y_val = generate_batches(
        adj_val, features_val, y_val, args.batch_size, args.use_master_node
    )

    adj_test = [adj[i] for i in test_index]
    features_test = [features[i] for i in test_index]
    y_test = [y[i] for i in test_index]
    adj_test, features_test, batch_n_graphs_test, y_test = generate_batches(
        adj_test, features_test, y_test, args.batch_size, args.use_master_node
    )
    print("=> Val + Test Loading Completed")
    print("=" * 50)
    # create model + optimizer
    model = MPAD(
        embeddings.shape[1],
        args.message_passing_layers,
        args.hidden,
        args.penultimate,
        nclass,
        args.dropout,
        embeddings,
        args.use_master_node,
        mp_type=args.mp_type,
    ).to("cuda:0")

    model.embedding_dim = args.hidden

    sim = BYOL(
        backbone=model, project_dim=args.hidden, bottle_neck_dim=args.bottleneck
    ).to("cuda:0")
    parameters = filter(lambda p: p.requires_grad, sim.parameters())

    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler.upper() == "COSINE":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=8, eta_min=1e-5, verbose=True
        )
    elif args.scheduler.upper() == "STEP":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        scheduler = None
    print("=> Model, Optimizer, and Scheduler Loading Complete")
    print("=" * 50)

    """
    MAIN TRAINING LOOP
    """
    model_path, model_id = make_model_id(args)

    if args.use_nlp_aug == False:
        aug_1 = aug_dict[args.rand_aug_1]
        aug_2 = aug_dict[args.rand_aug_2]

    best_epoch = 0.0
    best_ckpt = {}
    losses = collections.defaultdict(list)
    knn_acc = np.nan

    K = args.epochs * n_train_batches
    curr_k = 0.0
    for epoch in range(1, args.epochs):
        running_backbone_norm = 0.0
        running_backbone_std = 0.0
        running_encoder_norm = 0.0
        running_encoder_std = 0.0
        running_predictor_norm = 0.0
        running_predictor_std = 0.0

        if args.use_nlp_aug is False:
            adj_train_1 = aug_1(adj, mask=None, args=args)
            adj_train_2 = aug_2(adj, mask=None, args=args)

            # if args.normalize:
            #    adj_train_1 = [normalize(a) for a in adj_train_1]
            #    adj_train_2 = [normalize(a) for a in adj_train_2]

            features_train_1 = features_train_2 = features
            y_train_1 = y_train_2 = y_train
        elif args.use_nlp_aug:
            adj_train_1, features_train_1, y_train_1 = load_epoch_txt(
                epoch_num=epoch, l_enc=l_enc, args=args
            )
            adj_train_2, features_train_2, y_train_2 = load_epoch_txt(
                epoch_num=args.epochs - epoch, l_enc=l_enc, args=args
            )

            adj_train_1 = [adj_train_1[i] for i in train_index]
            features_train_1 = [features_train_1[i] for i in train_index]
            y_train_1 = [y_train_1[i] for i in train_index]

            adj_train_2 = [adj_train_2[i] for i in train_index]
            features_train_2 = [features_train_2[i] for i in train_index]
            y_train_2 = [y_train_2[i] for i in train_index]

        else:
            print("ERROR!!")
        (
            adj_train_1,
            features_train_1,
            batch_n_graphs_train_1,
            y_train_1,
        ) = generate_batches(
            adj_train_1,
            features_train_1,
            y_train_1,
            args.batch_size,
            args.use_master_node,
        )
        (
            adj_train_2,
            features_train_2,
            batch_n_graphs_train_2,
            y_train_2,
        ) = generate_batches(
            adj_train_2,
            features_train_2,
            y_train_2,
            args.batch_size,
            args.use_master_node,
        )
        sim.train()
        running_loss = 0.0
        for i in range(n_train_batches):

            a_1 = adj_train_1[i].to("cuda:0").to_dense().unsqueeze(0)
            a_2 = adj_train_2[i].to("cuda:0").to_dense().unsqueeze(0)
            f_1 = features_train_1[i].to("cuda:0")
            f_2 = features_train_2[i].to("cuda:0")
            b_n = batch_n_graphs_train_1[i].to("cuda:0")
            loss = train_simsiam(
                sim,
                optimizer,
                epoch,
                a_1,
                f_1,
                b_n,
                a_2,
                f_2,
            )
            curr_k += 1
            sim.update_moving_average(global_step=curr_k, max_steps=K)
            running_loss += loss.item()

            # iteration level book-keeping!
            sim.eval()
            with torch.no_grad():

                a_1 = adj_train_1[i].to("cuda:0").to_dense().unsqueeze(0)
                a_2 = adj_train_2[i].to("cuda:0").to_dense().unsqueeze(0)
                f_1 = features_train_1[i].to("cuda:0")
                f_2 = features_train_2[i].to("cuda:0")
                b_n = batch_n_graphs_train_1[i].to("cuda:0")
                # MPAD Backbone
                _, b1 = sim.encoder[0](
                    f_1,
                    a_1,
                    b_n,
                )
                _, b2 = sim.encoder[0](f_2, a_2, b_n)

                # Intermedate batchnorm
                p1 = sim.encoder[1](b1)
                p2 = sim.encoder[1](b2)

                # Predictor (projector is removed!)
                z1 = sim.online_predictor(p1)
                z2 = sim.online_predictor(p2)

                # similarities
                b_sim = F.cosine_similarity(b1, b2, dim=1).mean()
                p_sim = F.cosine_similarity(p1, p2, dim=1).mean()
                z_sim = F.cosine_similarity(z1, z2, dim=1).mean()

                # norms & std
                running_backbone_norm += b1.norm(dim=1).mean()
                running_backbone_std += (
                    (b1 / b1.norm(dim=1, keepdim=True) + 1e-10).std(dim=0).mean()
                )

                running_encoder_norm += p1.norm(dim=1).mean()
                running_encoder_std += (
                    (p1 / p1.norm(dim=1, keepdim=True) + 1e-10).std(dim=0).mean()
                )

                running_predictor_norm += z1.norm(dim=1).mean()
                running_predictor_std += (
                    (z1 / z1.norm(dim=1, keepdim=True) + 1e-10).std(dim=0).mean()
                )

                losses["backbone_sim"].append(b_sim.item())
                losses["encoder_sim"].append(p_sim.item())
                losses["predictor_sim"].append(z_sim.item())
            sim.train()

        """
        Epoch Level Bookkeeping!
        """

        if scheduler is not None:
            scheduler.step()
        # KNN Accuracy!
        sim.eval()

        val_embeds = []
        for i in range(n_val_batches):
            with torch.no_grad():
                a = adj_val[i].to("cuda:0").to_dense().unsqueeze(0)
                f = features_val[i].to("cuda:0")
                b = batch_n_graphs_val[i].to("cuda:0")
                val_embeds += sim.encoder[0](f, a, b)[1]
        val_embeds = torch.stack(val_embeds).cpu().numpy()

        test_embeds = []
        for i in range(n_test_batches):
            with torch.no_grad():
                a = adj_test[i].to("cuda:0").to_dense().unsqueeze(0)
                f = features_test[i].to("cuda:0")
                b = batch_n_graphs_test[i].to("cuda:0")
                test_embeds += sim.encoder[0](f, a, b)[1]
        test_embeds = torch.stack(test_embeds).cpu().numpy()
        knn_acc = knn(val_embeds, torch.cat(y_val), test_embeds, torch.cat(y_test))

        running_loss /= n_train_batches
        running_backbone_norm /= n_train_batches
        running_backbone_std /= n_train_batches
        running_encoder_norm /= n_train_batches
        running_encoder_std /= n_train_batches
        running_predictor_norm /= n_train_batches
        running_predictor_std /= n_train_batches

        losses["epoch_loss"].append(running_loss)
        losses["knn_acc"].append(knn_acc)
        losses["backbone_norm"].append(running_backbone_norm)
        losses["backbone_std"].append(running_backbone_std)
        losses["encoder_norm"].append(running_encoder_norm)
        losses["encoder_std"].append(running_encoder_std)
        losses["predictor_norm"].append(running_predictor_norm)
        losses["predictor_std"].append(running_predictor_std)

        print("=" * 50)
        print(
            "Epoch: {e} -- Loss: {l:.4f} -- KNN Acc: {knn:.4f} -- BBS: {bbn:.4f} BNS:{bns:.4f} PS: {preds:.4f}".format(
                e=epoch,
                l=losses["epoch_loss"][-1],
                knn=knn_acc,
                bbn=b_sim.item(),
                bns=p_sim.item(),
                preds=z_sim.item(),
            )
        )
        print("=" * 50)
        print()
        ckpt = {}
        ckpt["epoch"] = i
        ckpt["knn_acc"] = knn_acc
        ckpt["model"] = sim.state_dict()
        ckpt["args"] = vars(args)
        ckpt["stats"] = losses
        # torch.save(
        #    ckpt,
        #    "{model_path}/epoch_{e}.pth".format(
        #        model_path=model_path,
        #        model_id=model_id,
        #        e=epoch,
        #    ),
        # )
        if knn_acc > best_epoch:
            ckpt["unsupervised_best_epoch"] = epoch
            best_epoch = knn_acc
            #torch.save(
            #    best_ckpt,
            #    "{model_path}/best.pth".format(
            #        model_path=model_path,
            #        model_id=model_id,
            #    ),
            #)
            print("Epoch {0} -- Best Acc: {1:.4f} ".format(epoch, best_epoch))

    print("*=" * 50)
    print("Mean KNN ACC: ", torch.Tensor(losses["knn_acc"][-5:]).mean())
    print("Max KNN ACC: ", torch.Tensor(losses["knn_acc"]).max())
    print("*=" * 50)

    final_ckpt = {}
    # final_ckpt["knn_acc"] = knn_acc
    # final_ckpt["net"] = sim.state_dict()
    # final_ckpt["args"] = vars(args)
    # final_ckpt["stats"] = losses
    # torch.save(
    #     final_ckpt,
    #     "{model_path}/final.pth".format(model_path=model_path, model_id=model_id),
    # )

    log_str = "byol_{id}\t{acc}\n".format(
        id=model_id,
        acc=torch.Tensor(losses["knn_acc"][-5:]).mean(),
    )
    with open("mp_log.tsv", "a+") as f:
        f.write(log_str)
    return sim, losses, final_ckpt


if __name__ == "__main__":
    main()
