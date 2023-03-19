import torch
import torch.backends.cudnn as cudnn
import torch_geometric as geom

torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False

import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable
from copy import deepcopy
import pdb
from math import pi, cos 

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


class multiSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

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
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class MNIST_GNN(torch.nn.Module):
    def __init__(self, args):
        super(MNIST_GNN, self).__init__()

        self.embedding_h = torch.nn.Linear(args.input_dim, args.hidden_dim)

        self.ginlayers = torch.nn.ModuleList()

        for layer in range(args.num_layers):
            mlp = MLP(
                args.num_mlp_layers, args.hidden_dim, args.hidden_dim, args.hidden_dim
            )
            self.ginlayers.append(geom.nn.GINConv(mlp))

        #self.linears_prediction = torch.nn.ModuleList()
        self.args = args
        self.num_layers = args.num_layers

        if args.supervised == True:
            self.supervised = True
            for layer in range(args.num_layers + 1):
                self.linears_prediction.append(
                    nn.Linear(args.hidden_dim, args.num_classes)
                )
        else:
            self.supervised = False

        if self.args.readout == "sum":
            self.pool = geom.nn.global_add_pool
        elif self.args.readout == "mean":
            self.pool = geom.nn.global_mean_pool
        elif args.readout == "max":
            self.pool = geom.nn.global_max_pool
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, batch):

        hidden_rep = []
        x = self.embedding_h(x)
        hidden_rep.append(self.pool(x, batch))

        for i in range(self.args.num_layers):
            x = self.ginlayers[i](x, edge_index)
            hidden_rep.append(self.pool(x, batch))
        if self.supervised:
            score_over_layer = 0
            for i, x_p in enumerate(hidden_rep):
                score_over_layer += self.linears_prediction[i](x_p)
            return x, score_over_layer
        x = torch.cat(hidden_rep, dim=1)
        return x

class BYOL(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        self.project_dim = args.project_dim
        self.predictor_dim = args.predictor_dim
        self.bottle_neck_dim = args.bottle_neck_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.backbone=backbone
        if args.intermediate_bn:
            self.bn_int = torch.nn.BatchNorm1d(self.hidden_dim * (self.num_layers + 1))
            self.encoder = multiSequential(self.backbone, self.bn_int)
            self.target_encoder = multiSequential(deepcopy(self.backbone),deepcopy(self.bn_int))
        else:
            args.bn_int = None
            self.encoder = multiSequential(backbone)
            self.target_encoder = multiSequential(deepcopy(self.backbone))

        self.predictor_dim = self.hidden_dim * ((self.num_layers) +1)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(self.predictor_dim, self.bottle_neck_dim, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.bottle_neck_dim, self.predictor_dim),
        )

    
    def target_ema(self, k, K, base_ema=4e-3):
        # tau_base = 0.996 
        # base_ema = 1 - tau_base = 0.996 
        return 1 - base_ema * (cos(pi*k/K)+1)/2 
        # return 1 - (1-self.tau_base) * (cos(pi*k/K)+1)/2 

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
            
    def forward(self, data_1, data_2):
        f_o, h_o = self.encoder, self.predictor
        f_t      = self.target_encoder

        z1_o = f_o(data_1.x, data_1.edge_index, data_1.batch)     
        z2_o = f_o(data_2.x, data_2.edge_index, data_2.batch)     

        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)

        with torch.no_grad():
            z1_t = f_t(data_1.x, data_1.edge_index, data_1.batch)     
            z2_t = f_t(data_2.x, data_2.edge_index, data_2.batch)     
        
        L = self.D(p1_o, z2_t) / 2 + self.D(p2_o, z1_t) / 2 
        return L

    def D(self, p, z):
        return -torch.nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()

class SimCLR(nn.Module):
    def __init__(self, backbone, args):  # ,out_dim=60
        super().__init__()

        self.args = args
        self.project_dim = args.project_dim
        self.predictor_dim = args.predictor_dim
        self.bottle_neck_dim = args.bottle_neck_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        if args.intermediate_bn:
            self.bn_int = torch.nn.BatchNorm1d(self.hidden_dim * (self.num_layers + 1))
            self.encoder = multiSequential(self.backbone, self.bn_int)

        else:
            args.bn_int = None
            self.encoder = multiSequential(backbone)

        self.predictor_dim = self.hidden_dim * ((self.num_layers) +1)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(self.predictor_dim, self.bottle_neck_dim, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.bottle_neck_dim, self.predictor_dim),
        )

    def forward(self, data_1, data_2):
        f, h = self.encoder, self.predictor
        z1 = f(data_1.x, data_1.edge_index, data_1.batch)
        z2 = f(data_2.x, data_2.edge_index, data_2.batch)
        p1, p2 = h(z1), h(z2)
        L = self.NT_XentLoss(p1,p2) 
        return L

    def NT_XentLoss(self,z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape 
        device = z1.device 
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2*N, dtype=torch.bool, device=device)
        diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

        negatives = similarity_matrix[~diag].view(2*N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature

        labels = torch.zeros(2*N, device=device, dtype=torch.int64)

        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)



class SimSiam(nn.Module):
    def __init__(self, backbone, args):  # ,out_dim=60
        super().__init__()

        self.args = args
        self.project_dim = args.project_dim
        self.predictor_dim = args.predictor_dim
        self.bottle_neck_dim = args.bottle_neck_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(
                self.hidden_dim * (self.num_layers + 1),
                self.project_dim,
                bias=False,
            ),
            torch.nn.BatchNorm1d(self.project_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.project_dim, self.predictor_dim, bias=False),
            torch.nn.BatchNorm1d(self.predictor_dim),
        )

        if args.intermediate_bn:
            self.bn_int = torch.nn.BatchNorm1d(self.hidden_dim * (self.num_layers + 1))
            self.encoder = multiSequential(self.backbone, self.bn_int, self.projector)

        else:
            args.bn_int = None
            self.encoder = multiSequential(backbone, self.projector)

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(self.predictor_dim, self.bottle_neck_dim, bias=False),
            torch.nn.BatchNorm1d(self.bottle_neck_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.bottle_neck_dim, self.predictor_dim),
        )

    def forward(self, data_1, data_2):
        f, h = self.encoder, self.predictor
        z1 = f(data_1.x, data_1.edge_index, data_1.batch)
        z2 = f(data_2.x, data_2.edge_index, data_2.batch)
        p1, p2 = h(z1), h(z2)
        L = self.D_simplified(p1, z2) / 2 + self.D_simplified(p2, z1) / 2
        return L

    def D_simplified(self, p, z):
        return -torch.nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()
