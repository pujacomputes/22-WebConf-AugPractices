import torch
import numpy as np
import pdb
from scipy import sparse


def rand_node_augment(adj, mask, args):
    # rancdom aug
    a_prime_list = []
    for a in adj:
        a_prime = a.toarray()
        clipping_num = int(np.floor(float(args.rand_node_drop) * a_prime.shape[0]))
        r_idx = np.random.randint(
            low=0, high=a_prime.shape[0], size=clipping_num, dtype=int
        )
        for tmp_idx in r_idx:
            a_prime[tmp_idx, :] = 0
            a_prime[:, tmp_idx] = 0
        a_prime_list.append(sparse.csr_matrix(a_prime))

    return a_prime_list


def rand_edge_augment(adj, mask, args):
    a_prime_list = []
    # rancdom aug
    for a in adj:
        a_prime = a.toarray()
        clipping_num = int(np.floor(float(args.rand_edge_perturb) * a_prime.shape[0]))
        
        #get number of edges to clip!
        r_idx = np.random.randint(
            low=0, high=a_prime.shape[0], size=(clipping_num,2), dtype=int
        )

        for e_num, (u,v) in enumerate(r_idx):
            # if edge exists drop it.
            if a_prime[u, v] >= 0.5 or a_prime[v, u] >= 0.5:
                a_prime[u, v] = 0
                a_prime[v, u] = 0

            # if edge does not exist, create it.
            elif a_prime[u, v] <= 0.5 or a_prime[v, u] <= 0.5:
                val = np.random.random_sample()
                a_prime[u, v] = val
                a_prime[v, u] = val

            else:
                print("ERROR",a_prime[u,v],a_prime[v,u])
        a_prime_list.append(sparse.csr_matrix(a_prime))
    return a_prime_list

def rand_edge_augment_real(adj, mask, args):
    a_prime_list = []
    # rancdom aug
    for a in adj:
        a_prime = a.toarray()
        clipping_num = int(np.floor(float(args.rand_edge_perturb) * a_prime.shape[0]))
        
        #get number of edges to clip!
        r_idx = np.random.randint(
            low=0, high=a_prime.shape[0], size=(clipping_num,2), dtype=int
        )

        for e_num, (u,v) in enumerate(r_idx):
            # if edge exists drop it.
            if a_prime[u, v] == 1 or a_prime[v, u] == 1:
                a_prime[u, v] = 0
                a_prime[v, u] = 0

            # if edge does not exist, create it.
            elif a_prime[u, v] == 0 or a_prime[v, u] == 0:
                a_prime[u, v] = 1
                a_prime[v, u] = 1

            else:
                print("ERROR")
        a_prime_list.append(sparse.csr_matrix(a_prime))
    return a_prime_list

def rand_subgraph_augment(adj, mask, args):

    """
    This augmentation masks out the subgraph found by random walking!
    It does not use it as an alternative view!!
    """
    # for each graph in subgraph select a starting node
    a_prime_list = []
    for a in adj:
        a_prime = a.toarray()
 
        valid_start= a_prime.shape[0]

        starting_idx = np.random.randint(low=0, high=valid_start, dtype=int)

        max_walk_len = int(np.ceil(float(args.rand_subgraph_drop) * a_prime.sum())) 
        avg_walk_len = 0
        r_walk = [starting_idx]
        count = 0
        diag = np.diagonal(a_prime)
        np.fill_diagonal(a_prime,(0))
        while len(r_walk) <= max_walk_len:
            if count > valid_start:
                break
            # pull out the neighbhors of last visited node
            neighbors = np.nonzero(a_prime[r_walk[-1]])[0]
            if len(neighbors) == 0:
                break
            n_idx = np.random.randint(low=0, high=len(neighbors))
            r_walk.append(neighbors[n_idx])
        # mask out this subgraph, given r_walk > 1
        if len(r_walk) > 1:
            avg_walk_len += len(r_walk)
            for i, j in zip(r_walk[0:], r_walk[1:]):
                a_prime[i, j] = 0
                if not args.directed:
                    a_prime[j, i] = 0

        np.fill_diagonal(a_prime,diag)
        a_prime_list.append(sparse.csr_matrix(a_prime))

    return a_prime_list


def rand_subgraph_augment_real(input_1, mask, args):

    """
    This augmentation masks out the subgraph found by random walking!
    It does not use it as an alternative view!!
    """
    # for each graph in subgraph select a starting node
    batch_size = input_1.shape[0]
    pert_input_1 = input_1.clone()
    valid_start = [m.sum().int() for m in mask]

    starting_idx = [
        torch.randint(low=0, high=m_size, size=(1,), dtype=torch.int)
        for m_size in valid_start
    ]

    max_walk_len = [
        (float(args.rand_subgraph_drop) * adj.sum()).ceil().int() for adj in input_1
    ]

    avg_walk_len = 0
    for b_idx in range(batch_size):
        r_walk = [starting_idx[b_idx].long().item()]
        count = 0
        pert_input_1[b_idx].fill_diagonal_(0)
        while len(r_walk) <= max_walk_len[b_idx]:
            if count > valid_start[b_idx]:
                break
            # pull out the neighbhors of last visited node
            neighbors = torch.nonzero(pert_input_1[b_idx][r_walk[-1]])
            if len(neighbors) == 0:
                break
            n_idx = torch.randint(low=0, high=len(neighbors), size=(1,))
            r_walk.append(neighbors[n_idx].long().item())
        # mask out this subgraph, given r_walk > 1
        if len(r_walk) > 1:
            avg_walk_len += len(r_walk)
            for i, j in zip(r_walk[0:], r_walk[1:]):
                pert_input_1[b_idx][i, j] = 0
                pert_input_1[b_idx][j, i] = 0

    print("Avg Walk: ", avg_walk_len / batch_size)
    mask = torch.eye(input_1.shape[1]).to(args["SHARED"]["device"]) * input_1
    pert_input_1 += mask
    return pert_input_1
