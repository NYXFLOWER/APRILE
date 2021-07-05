from sklearn import metrics
from itertools import combinations
from itertools import chain, product
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch

def negative_sampling(pos_edge_index, num_nodes):
    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    perm = torch.tensor(np.random.choice(num_nodes**2, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(np.random.choice(num_nodes**2, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).long().to(pos_edge_index.device)


def typed_negative_sampling(pos_edge_index, num_nodes, range_list):
    tmp = []
    for start, end in range_list:
        tmp.append(negative_sampling(pos_edge_index[:, start: end], num_nodes))
    return torch.cat(tmp, dim=1)

    
def normalize(input):
    norm_square = (input ** 2).sum(dim=1)
    return input / torch.sqrt(norm_square.view(-1, 1))


def remove_bidirection(edge_index, edge_type):
    mask = edge_index[0] > edge_index[1]
    keep_set = mask.nonzero().view(-1)

    if edge_type is None:
        return edge_index[:, keep_set]
    else:
        return edge_index[:, keep_set], edge_type[keep_set]


def to_bidirection(edge_index, edge_type=None):
    tmp = edge_index.clone()
    tmp[0, :], tmp[1, :] = edge_index[1, :], edge_index[0, :]
    if edge_type is None:
        return torch.cat([edge_index, tmp], dim=1)
    else:
        return torch.cat([edge_index, tmp], dim=1), torch.cat(
            [edge_type, edge_type])


def get_range_list(edge_list):
    tmp = []
    s = 0
    for i in edge_list:
        tmp.append((s, s + i.shape[1]))
        s += i.shape[1]
    return torch.tensor(tmp)


def process_edges(raw_edge_list, p=0.9):
    train_list = []
    test_list = []
    train_label_list = []
    test_label_list = []

    for i, idx in enumerate(raw_edge_list):
        train_mask = np.random.binomial(1, p, idx.shape[1])
        test_mask = 1 - train_mask
        train_set = train_mask.nonzero()[0]
        test_set = test_mask.nonzero()[0]

        train_list.append(idx[:, train_set])
        test_list.append(idx[:, test_set])

        train_label_list.append(
            torch.ones(2 * train_set.size, dtype=torch.long) * i)
        test_label_list.append(
            torch.ones(2 * test_set.size, dtype=torch.long) * i)

    train_list = [to_bidirection(idx) for idx in train_list]
    test_list = [to_bidirection(idx) for idx in test_list]

    train_range = get_range_list(train_list)
    test_range = get_range_list(test_list)

    train_edge_idx = torch.cat(train_list, dim=1)
    test_edge_idx = torch.cat(test_list, dim=1)

    train_et = torch.cat(train_label_list)
    test_et = torch.cat(test_label_list)

    return train_edge_idx, train_et, train_range, test_edge_idx, test_et, test_range


def sparse_id(n):
    idx = [[i for i in range(n)], [i for i in range(n)]]
    val = [1 for i in range(n)]
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(val)
    shape = (n, n)

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def dense_id(n):
    idx = [i for i in range(n)]
    val = [1 for _ in range(n)]
    out = sp.coo_matrix((val, (idx, idx)), shape=(n, n), dtype=float)

    return torch.Tensor(out.todense())


def auprc_auroc_ap(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    auroc, ap = metrics.roc_auc_score(y, pred), metrics.average_precision_score(y, pred)
    y, xx, _ = metrics.precision_recall_curve(y, pred)
    auprc = metrics.auc(xx, y)

    return auprc, auroc, ap


def uniform(size, tensor):
    bound = 1.0 / np.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def dict_ep_to_nparray(out_dict, epoch):
    out = np.zeros(shape=(3, epoch))
    for ep, [prc, roc, ap] in out_dict.items():
        out[0, ep] = prc
        out[1, ep] = roc
        out[2, ep] = ap
    return out


def get_indices_mask(indices, in_indices):
    d = indices.shape[-1]
    isin = np.isin(indices, in_indices).reshape(-1, d)
    mask = isin.all(axis=0)
    return torch.from_numpy(mask)


def get_edge_index_from_coo(mat, bidirection):
    """convert sparse.coo to sparse torch.Tensor

    Args:
        mat (sparse.coo): a sparse adjacency matrix
        bidirection (bool): if use undirected edge

    Returns:
        torch.Tensor: shape (2, n_edge)
    """
    if bidirection:
        mask = mat.row > mat.col
        half = np.concatenate(
            [mat.row[mask].reshape(1, -1), mat.col[mask].reshape(1, -1)],
            axis=0)
        full = np.concatenate([half, half[[1, 0], :]], axis=1)
        return torch.from_numpy(full.astype(np.int64))
    else:
        tmp = np.concatenate([mat.row.reshape(1, -1), mat.col.reshape(1, -1)],
                             axis=0)
        return torch.from_numpy(tmp.astype(np.int64))


def visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, pp_adj, d1, d2,
                    out_path,
                    protein_name_dict=None, drug_name_dict=None, hiden=True,
                    size=(40, 40)):
    """visualize Aprile-Exp's outputs
        1. use different color for pp and pd edges
        2. annotate the weight of each edge near the edge (or annotate with the tranparentness of edges for each edge)
        3. annotate the name of each node near the node, if name_dict=None, then annotate with node's index

    Args:
        pp_idx (torch.Tensor): integer tensor (2, n_pp_edges)
        pp_weight (torch.Tensor): float tensor (1, n_pp_edges), values with (0, 1)
        pd_idx (torch.Tensor): integer tensor (2, n_pd_edges)
        pd_weight (torch.Tensor): float tensor (1, n_pd_edges), values with (0, 1)
        pp_adj ([type]): [description]
        d1 (list): drug list
        d2 (list): drug list pairing with `d1`
        out_path (str): output path
        protein_name_dict (dict, optional): the mapping for protein makers' text. Defaults to None.
        drug_name_dict (dict, optional): the mapping for drug markers' text. Defaults to None.
        hiden (bool, optional): if show related edge with edge weight of 0.01. Defaults to True.
        size (tuple, optional): the figure size. Defaults to (40, 40).

    Returns:
        networkx.Graph: the graph object
        matplotlib.pyplot.figure: the ploted figure
    """

    G = nx.Graph()
    pp_edge, pd_edge, pp_link = [], [], []
    p_node, d_node = set(), set()

    if not protein_name_dict:
        tmp = set(pp_idx.flatten()) | set(pd_idx[0])
        protein_name_dict = {i: 'p-' + str(i) for i in tmp}
    if not drug_name_dict:
        drug_name_dict = {i: 'd-' + str(i) for i in set(pd_idx[1])}

    # add pp edges
    for e in zip(pp_idx.T, pp_weight.T):
        t1, t2 = protein_name_dict[e[0][0]], protein_name_dict[e[0][1]]
        G.add_edge(t1, t2, weights=e[1])
        pp_edge.append((t1, t2))
        p_node.update([t1, t2])

    # add pd edges
    for e in zip(pd_idx.T, pd_weight.T):
        t1, t2 = protein_name_dict[e[0][0]], drug_name_dict[e[0][1]]
        G.add_edge(t1, t2, weights=e[1])
        pd_edge.append((t1, t2))
        p_node.add(t1)
        d_node.add(t2)

    # add dd edges
    dd_edge = []
    for e in zip(d1, d2):
        t1, t2 = drug_name_dict[int(e[0])], drug_name_dict[int(e[1])]
        G.add_edge(t1, t2, weights=999)
        dd_edge.append((t1, t2))

    if hiden:
        # add underline pp edges
        pp_edge_idx = pp_idx.tolist()
        pp_edge_idx = set(zip(pp_edge_idx[0], pp_edge_idx[1]))
        p_node_idx = list(set(pp_idx.flatten().tolist()))
        pp_adj_idx = pp_adj.tolist()
        pp_adj_idx = set(zip(pp_adj_idx[0], pp_adj_idx[1]))

        combins = [c for c in combinations(p_node_idx, 2)]
        for i, j in combins:
            if (i, j) in pp_adj_idx or (j, i) in pp_adj_idx:
                if (i, j) not in pp_edge_idx and (j, i) not in pp_edge_idx:
                    G.add_edge(protein_name_dict[i], protein_name_dict[j],
                               weights='0')
                    pp_link.append((protein_name_dict[i], protein_name_dict[j]))
        print(len(pp_link))
    # draw figure
    plt.figure(figsize=size)

    # draw nodes
    pos = nx.spring_layout(G)
    for p in d_node:  # raise drug nodes positions
        pos[p][1] += 1
    nx.draw_networkx_nodes(G, pos, nodelist=p_node, node_size=500,
                           node_color='y')
    nx.draw_networkx_nodes(G, pos, nodelist=d_node, node_size=500,
                           node_color='blue')

    # draw edges and edge labels
    nx.draw_networkx_edges(G, pos, edgelist=pp_edge, width=2)
    if hiden:
        nx.draw_networkx_edges(G, pos, edgelist=pp_link, width=2,
                               edge_color='gray', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=pd_edge, width=2, edge_color='g')
    nx.draw_networkx_edges(G, pos, edgelist=dd_edge, width=2,
                           edge_color='black', alpha=0.5)

    nx.draw_networkx_edge_labels(G, pos, font_size=10,
                                 edge_labels={(u, v): str(d['weights'])[:4] for
                                              u, v, d in G.edges(data=True)})

    # draw node labels
    for p in pos:  # raise text positions
        pos[p][1] += 0.02
    nx.draw_networkx_labels(G, pos, font_size=14)
    if out_path is not None:
        plt.savefig(out_path)
        print(f"DONE --> save figure to path: \"{out_path}\" ")
    
    return G, plt.gcf()


def args_parse_train(drug_index_1, drug_index_2, side_effect_index, rg, et, idx):
    """
    :param drug_index_1: char '*' or string of the format list of int, like 2,3,4
    :param drug_index_2: char '*' or string of the format list of int
    :param side_effect_index: char '*' or string of the format list of int
    :param rg: int tensor of shape (n_side_effect, 2)
    :param et: int tensor of shape (n_dd_edge)
    :param idx: int tensor of shape (2, n_dd_edge)
    :return: three lists of int
    """

    if drug_index_1 != 'all':
        drug_index_1 = [int(i) for i in drug_index_1.split(',')]
    if drug_index_2 != 'all':
        drug_index_2 = [int(i) for i in drug_index_2.split(',')]
    if side_effect_index != 'all':
        side_effect_index = [int(i) for i in side_effect_index.split(',')]
        # side_effect_index = [list(range(rg[i][0], rg[i][1])) for i in side_effect_index]
        # side_effect_index = list(chain(*side_effect_index))

    # case - * * *
    if drug_index_1 == 'all' and drug_index_2 == 'all' and side_effect_index == 'all':
        return idx[0].tolist(), idx[1].tolist(), et.tolist()

    if isinstance(drug_index_1, list) and isinstance(drug_index_2, list):
        drug1, drug2, side_effect = [], [], []
        # case - [] [] []
        if isinstance(drug_index_2, list) and isinstance(side_effect_index, list):
            for s, d1, d2 in product(side_effect_index, drug_index_1, drug_index_2):
                d1_d2 = idx[:, rg[s, 0]:rg[s, 1]]
                d1, d2 = (d1, d2) if d1 < d2 else (d2, d1)
                if d2 in d1_d2[1][d1_d2[0] == d1]:
                    drug1.append(d1)
                    drug2.append(d2)
                    side_effect.append(s)
        # case - [] [] *
        else:
            for d1, d2 in product(drug_index_1, drug_index_2):
                d1, d2 = (d1, d2) if d1 < d2 else (d2, d1)
                d1_d2 = idx[1][idx[0] == d1]
                if d2 in d1_d2:
                    tmp = et[idx[0] == d1]
                    tmp = (tmp[d1_d2 == d2]).tolist()
                    side_effect.extend(tmp)
                    drug1.extend([d1] * len(tmp))
                    drug2.extend([d2] * len(tmp))
        return drug1, drug2, side_effect

    if isinstance(side_effect_index, list):
        # case - * * []
        et_index = [list(range(rg[i][0], rg[i][1])) for i in side_effect_index]
        et_index = list(chain(*et_index))
        drug1, drug2, side_effect = idx[0][et_index], idx[1][et_index], et[et_index]

        # case - * [] [] or [] * []
        if isinstance(drug_index_1, list) or isinstance(drug_index_2, list):
            drug_index_1, idrug1, idrug2 = (drug_index_1, drug1, drug2) \
                if isinstance(drug_index_1, list) \
                else (drug_index_2, drug2, drug1)
            iside_effect = side_effect
            drug1, drug2, side_effect = [], [], []
            for d1 in drug_index_1:
                tmp = (idrug2[idrug1==d1]).tolist()
                drug2.extend(tmp)
                side_effect.extend((iside_effect[idrug1==d1]).tolist())
                drug1.extend([d1] * len(tmp))
            return drug1, drug2, side_effect

        drug1, drug2, side_effect = drug1.tolist(), drug2.tolist(), side_effect.tolist()
    else:
        # case - [] * * or * [] *
        drug_index_1 = drug_index_1 if isinstance(drug_index_1, list) else drug_index_2
        drug1, drug2, side_effect = [], [], []
        for d1 in drug_index_1:
            tmp = (idx[1][idx[0] == d1]).tolist()
            drug2.extend(tmp)
            side_effect.extend((et[idx[0] == d1]).tolist())
            drug1.extend([d1] * len(tmp))

    return drug1, drug2, side_effect


def args_parse_pred(drug_index_1, drug_index_2, side_effect_index, n_drug, n_side_effect):
    """
    :param drug_index_1: char '*' or string of the format list of int, like 2,3,4
    :param drug_index_2: char '*' or string of the format list of int
    :param side_effect_index: char '*' or string of the format list of int
    :return: three lists of int
    """
    # case - * * *
    if drug_index_1 == 'all' and drug_index_2 == 'all' and side_effect_index == 'all':
        print('The drug1, drug2, side effect inputs can not be [all] at the same time!')
        exit()
        return

    drug_index_1 = list(range(n_drug)) if drug_index_1 == 'all' \
        else [int(i) for i in drug_index_1.split(',')]
    drug_index_2 = list(range(n_drug)) if drug_index_2 == 'all' \
        else [int(i) for i in drug_index_2.split(',')]
    side_effect_index = list(range(n_side_effect)) if side_effect_index == 'all' \
        else [int(i) for i in side_effect_index.split(',')]

    drug1, drug2, side_effect = [], [], []
    for s, d1, d2 in product(side_effect_index, drug_index_1, drug_index_2):
        if d1 != d2:
            side_effect.extend([s, s])
            drug1.extend((d1, d2))
            drug2.extend((d2, d1))

    return drug1, drug2, side_effect
