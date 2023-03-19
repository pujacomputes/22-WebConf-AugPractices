import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MessagePassing, Attention
import copy
from math import pi, cos
from collections import OrderedDict
from torch_geometric.nn import (
    DenseGCNConv,
    DenseGINConv,
    DenseGraphConv,
    DenseSAGEConv,
)
import pdb


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        device: which device to use
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                # h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                h = self.linears[layer](h)

                if h.dim() == 3:
                    batch_num, num_nodes, channels = h.shape
                    h = self.batch_norms[layer](h.permute(0, 2, 1)).permute(0, 2, 1)
                    h = F.relu(h)

                elif h.dim() == 2:
                    num_nodes, channels = h.shape
                    h = self.batch_norms[layer](h)
                    h = F.relu(h)
            if h.dim() == 3:
                h = h.squeeze(0)
            return self.linears[self.num_layers - 1](h)


class GCNMessagePassing(torch.nn.Module):
    """
    Simple Message Passing layer
    """

    def __init__(self, in_features, out_features, mp_type="gcn"):
        super(GCNMessagePassing, self).__init__()
        self.mlp1 = MLP(2, in_features, out_features, out_features)
        self.mlp2 = MLP(2, out_features, out_features, out_features)

        self.fc1_update = nn.Linear(out_features, out_features)
        self.fc2_update = nn.Linear(out_features, out_features)
        self.fc1_reset = nn.Linear(out_features, out_features)
        self.fc2_reset = nn.Linear(out_features, out_features)
        self.fc1 = nn.Linear(out_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

        if mp_type == "gcn":
            print("Using => GCN")
            self.message_passing = DenseGCNConv(
                in_channels=out_features, out_channels=out_features
            )
        elif mp_type == "gin":
            print("Using => GIN")
            nn_mlp = torch.nn.Sequential(
                torch.nn.Linear(out_features, out_features),
                #torch.nn.BatchNorm1d(out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(out_features, out_features),
                torch.nn.ReLU(),
            )
            self.message_passing = DenseGINConv(nn_mlp)
        elif mp_type == "graph":
            print("Using => GRAPH")
            self.message_passing = DenseGraphConv(
                in_channels=out_features, out_channels=out_features
            )
        elif mp_type == "sage":
            print("Using => SAGE")
            self.message_passing = DenseSAGEConv(
                in_channels=out_features, out_channels=out_features
            )
        else:
            print("ERROR NOT DEFINED!")

    def forward(self, x_in, adj):
        x = self.mlp1(x_in)

        out = self.message_passing(x=x, adj=adj)
        out = self.mlp2(out)

        z = torch.sigmoid(self.fc1_update(out) + self.fc2_update(x))
        r = torch.sigmoid(self.fc1_reset(out) + self.fc2_reset(x))
        out = torch.tanh(self.fc1(out) + self.fc2(r * x))
        out = (1 - z) * x + z * out
        return out


class MPAD(nn.Module):
    def __init__(
        self,
        n_feat,
        n_message_passing,
        n_hid,
        n_penultimate,
        n_class,
        dropout,
        embeddings,
        use_master_node,
        mp_type="gcn",
    ):
        super(MPAD, self).__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.n_message_passing = n_message_passing
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1]).to(
        device
        )
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = False
        self.embedding.to(device)

        self.mps = torch.nn.ModuleList()
        self.atts = torch.nn.ModuleList()
        for i in range(n_message_passing):
            if i == 0:
                self.mps.append(GCNMessagePassing(n_feat, n_hid, mp_type))
                # self.mps.append(MessagePassing(n_feat, n_hid))
            else:
                self.mps.append(GCNMessagePassing(n_hid, n_hid, mp_type))
                # self.mps.append(MessagePassing(n_hid, n_hid))
            self.atts.append(Attention(n_hid, n_hid, use_master_node))

        if use_master_node:
            self.bn = nn.BatchNorm1d(2 * n_message_passing * n_hid)
            self.fc1 = nn.Linear(2 * n_message_passing * n_hid, n_penultimate)
        else:
            self.bn = nn.BatchNorm1d(n_message_passing * n_hid)
            self.fc1 = nn.Linear(n_message_passing * n_hid, n_penultimate)

        self.fc2 = nn.Linear(n_penultimate, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj, n_graphs):
        try:
            x = self.embedding(x)
        except:
            pdb.set_trace()
        x = self.dropout(x)
        lst = list()
        for i in range(self.n_message_passing):
            x = self.mps[i](x, adj)
            t = x.view(n_graphs[0], -1, x.size()[1])
            t = self.atts[i](t)
            lst.append(t)
        x = torch.cat(lst, 1)
        x = self.bn(x)
        embed = self.relu(self.fc1(x))
        x = self.dropout(embed)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), embed


def D(p, z, version="simplified"):  # negative cosine similarity
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


class SimSiam(torch.nn.Module):
    def __init__(self, backbone, project_dim, bottle_neck_dim):
        super().__init__()
        self.project_dim = project_dim
        self.embedding_dim = backbone.embedding_dim
        self.bottle_neck_dim = bottle_neck_dim

        self.backbone = backbone
        self.bn_int = torch.nn.BatchNorm1d(self.embedding_dim)

        self.encoder = nn.Sequential(self.backbone, self.bn_int)  # f encoder

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.bottle_neck_dim, bias=False),
            torch.nn.BatchNorm1d(self.bottle_neck_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.bottle_neck_dim, self.embedding_dim),
        )

    def forward(self, feat_1, adj_1, batch_n_graphs_1, feat_2, adj_2, batch_n_graphs_2):
        f, h = self.encoder, self.predictor
        _, z1 = f[0](feat_1, adj_1, batch_n_graphs_1)
        _, z2 = f[0](feat_2, adj_2, batch_n_graphs_2)

        z1 = f[1](z1)
        z2 = f[1](z2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return L


class SimCLR(torch.nn.Module):
    def __init__(self, backbone, project_dim, bottle_neck_dim):
        super().__init__()
        self.project_dim = project_dim
        self.embedding_dim = backbone.embedding_dim
        self.bottle_neck_dim = bottle_neck_dim

        self.backbone = backbone
        self.bn_int = torch.nn.BatchNorm1d(self.embedding_dim)

        self.encoder = nn.Sequential(self.backbone, self.bn_int)  # f encoder

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.bottle_neck_dim, bias=False),
            torch.nn.BatchNorm1d(self.bottle_neck_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.bottle_neck_dim, self.embedding_dim),
        )

    def NT_XentLoss(self, z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
        )
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

        negatives = similarity_matrix[~diag].view(2 * N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature

        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)

        loss = F.cross_entropy(logits, labels, reduction="sum")
        return loss / (2 * N)

    def forward(self, feat_1, adj_1, batch_n_graphs_1, feat_2, adj_2, batch_n_graphs_2):
        f, h = self.encoder, self.predictor
        _, z1 = f[0](feat_1, adj_1, batch_n_graphs_1)
        _, z2 = f[0](feat_2, adj_2, batch_n_graphs_2)

        z1 = f[1](z1)
        z2 = f[1](z2)
        p1, p2 = h(z1), h(z2)
        L = self.NT_XentLoss(p1, p2)
        return L


class BYOL(torch.nn.Module):
    def __init__(self, backbone, project_dim, bottle_neck_dim):
        super().__init__()
        self.project_dim = project_dim
        self.embedding_dim = backbone.embedding_dim
        self.bottle_neck_dim = bottle_neck_dim

        self.backbone = backbone
        self.bn_int = torch.nn.BatchNorm1d(self.embedding_dim)

        self.encoder = nn.Sequential(self.backbone, self.bn_int)  # f encoder
        self.target_encoder = copy.deepcopy(self.encoder)  # f encoder

        self.online_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.bottle_neck_dim, bias=False),
            torch.nn.BatchNorm1d(self.bottle_neck_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.bottle_neck_dim, self.embedding_dim),
        )

    def target_ema(self, k, K, base_ema=4e-3):
        # tau_base = 0.996
        # base_ema = 1 - tau_base = 0.996
        return 1 - base_ema * (cos(pi * k / K) + 1) / 2
        # return 1 - (1-self.tau_base) * (cos(pi*k/K)+1)/2

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        for online, target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data = tau * target.data + (1 - tau) * online.data

    def forward(self, feat_1, adj_1, batch_n_graphs_1, feat_2, adj_2, batch_n_graphs_2):
        f_o, h_o = self.encoder, self.online_predictor
        f_t = self.target_encoder

        _, z1_o = f_o[0](feat_1, adj_1, batch_n_graphs_1)
        _, z2_o = f_o[0](feat_2, adj_2, batch_n_graphs_2)

        z1_o = f_o[1](z1_o)
        z2_o = f_o[1](z2_o)

        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)

        with torch.no_grad():
            _, z1_t = f_t[0](feat_1, adj_1, batch_n_graphs_1)
            _, z2_t = f_t[0](feat_2, adj_2, batch_n_graphs_2)
            z1_t = f_t[1](z1_t)
            z2_t = f_t[1](z2_t)

        L = D(p1_o, z2_t) / 2 + D(p2_o, z1_t) / 2
        return L
