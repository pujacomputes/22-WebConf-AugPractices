import tqdm
import torch
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False

import pdb


def knn_monitor_gine(
    net,
    val_data_loader,
    test_data_loader,
    epoch,
    args,
    k=200,
    t=0.1,
    hide_progress=False,
):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    classes = args.label_dim
    with torch.no_grad():
        # generate feature bank
        targets = []
        for data, _ in tqdm.tqdm(
            val_data_loader,
            desc="Feature extracting",
            leave=False,
            disable=hide_progress,
        ):
            data.to(args.device)
            feature = net(data.x, data.edge_index, data.edge_attr, data.batch)
            feature = torch.nn.functional.normalize(feature, dim=1)
            feature_bank.append(feature)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(
            val_data_loader.dataset.label_all, device=args.device
        )
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm.tqdm(test_data_loader, desc="kNN", disable=hide_progress)
        for data, target in test_bar:
            data.to(args.device)
            feature = net(data.x, data.edge_index, data.edge_attr, data.batch)
            feature = torch.nn.functional.normalize(feature, dim=1)

            pred_labels = knn_predict(
                feature, feature_bank, feature_labels, classes, k, t
            )
            total_num += data.num_graphs
            total_top1 += (
                (pred_labels[:, 0] == target.to(args.device)).float().sum().item()
            )
            test_bar.set_postfix({"Accuracy": total_top1 / total_num * 100})
    return total_top1 / total_num * 100


def knn_monitor(
    net,
    val_data_loader,
    test_data_loader,
    epoch,
    args,
    k=200,
    t=0.1,
    hide_progress=False,
):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    classes = args.label_dim
    with torch.no_grad():
        # generate feature bank
        targets = []
        for data, _ in tqdm.tqdm(
            val_data_loader,
            desc="Feature extracting",
            leave=False,
            disable=hide_progress,
        ):
            data.to(args.device)
            feature = net(data.x, data.edge_index, data.batch)
            feature = torch.nn.functional.normalize(feature, dim=1)
            feature_bank.append(feature)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(
            val_data_loader.dataset.label_all, device=args.device
        )
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm.tqdm(test_data_loader, desc="kNN", disable=hide_progress)
        for data, target in test_bar:
            data.to(args.device)
            feature = net(data.x, data.edge_index, data.batch)
            feature = torch.nn.functional.normalize(feature, dim=1)

            pred_labels = knn_predict(
                feature, feature_bank, feature_labels, classes, k, t
            )
            total_num += data.num_graphs
            total_top1 += (
                (pred_labels[:, 0] == target.to(args.device)).float().sum().item()
            )
            test_bar.set_postfix({"Accuracy": total_top1 / total_num * 100})
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
