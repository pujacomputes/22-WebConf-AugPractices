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
from models import MPAD, SimSiam
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
        model_path = "GRAPH_CKPTS/"
        model_id = "_".join(
            [
                args.dataset,
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
        model_path = "NLP_CKPTS/"
        model_id = "_".join(
            [
                args.dataset,
                args.optim,
                str(args.hidden),
                str(args.bottleneck),
                str(args.batch_size),
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
    )

    model.embedding_dim = args.hidden

    sim = SimSiam(
        backbone=model, project_dim=args.hidden, bottle_neck_dim=args.bottleneck
    )

    sim.eval()
    accs = []
    for _ in range(10):
        val_embeds = []
        for i in range(n_val_batches):
            with torch.no_grad():
                val_embeds += sim.encoder[0](
                    features_val[i], adj_val[i], batch_n_graphs_val[i]
                )[1]
        val_embeds = torch.stack(val_embeds).numpy()

        test_embeds = []
        for i in range(n_test_batches):
            with torch.no_grad():
                test_embeds += sim.encoder[0](
                    features_test[i], adj_test[i], batch_n_graphs_test[i]
                )[1]
        test_embeds = torch.stack(test_embeds).numpy()
        knn_acc = knn(val_embeds, torch.cat(y_val), test_embeds, torch.cat(y_test))
        accs.append(knn_acc)
    accs = np.array(accs)
    print("Mean: ", np.mean(accs))
    print("Std: ", np.std(accs))


if __name__ == "__main__":
    main()
