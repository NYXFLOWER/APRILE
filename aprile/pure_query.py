import pickle
import os
import pandas
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_dict.pkl'), 'rb') as f:
    data = pickle.load(f)

class DataTmp(object):
    pass

gdata = DataTmp()
gdata.__dict__ = data

class PureAprileQuery(object):
    """For query I/O only
    """
    @staticmethod
    def load_from_pkl(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def __init__(self, drug1, drug2, side_effect, regularization=2):
        self.drug1 = drug1
        self.drug2 = drug2
        self.side_effect = side_effect
        self.regularization = regularization
        self.if_explain = False
        self.if_enrich = False
        self.if_pred = False

    def __repr__(self):
        return str(self.__class__) + ": \n" + str(self.__dict__)
        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def set_exp_result(self, pp_index, pp_weight, pd_index, pd_weight):
        if pd_index.shape[1]:
            self.pp_index = pp_index
            self.pp_weight = pp_weight
            self.pd_index = pd_index
            self.pd_weight = pd_weight
            self.if_explain = True

            print('pp_edge: {}, pd_edge:{}\n'.format(pp_index.shape[1], pd_index.shape[1]))

    def set_enrich_result(self, goea_results_sig):
        if len(goea_results_sig):
            self.if_enrich = True

            keys = ['name', 'namespace', 'id']
            df_go1 = pandas.DataFrame([{k: g.goterm.__dict__.get(k) for k in keys} for g in goea_results_sig])
            df_p = pandas.DataFrame([{'p_fdr_bh': g.__dict__['p_fdr_bh']} for g in goea_results_sig])
            df_go = df_go1.merge(df_p, left_index=True, right_index=True)

            go_genes = pandas.DataFrame([{'id': g.goterm.id, 'gene': s, 'symbol': gdata.geneid2symbol[s]} for g in goea_results_sig for s in g.study_items])
        
            self.GOEnrich_table = df_go.merge(go_genes, on='id')

    def set_pred_result(self, probability, piu_score, ppiu_score):
        self.probability = probability
        self.piu_score = piu_score
        self.ppiu_score = ppiu_score
        self.if_pred = True

    def get_query(self):
        return self.drug1, self.drug2, self.side_effect, self.regularization
        
    def get_pred_table(self):
        keys = ['drug_1', 'CID_1', 'name_1', 'drug_2', 'CID_2', 'name_2', 'side_effect', 'side_effect_name', 'prob', 'piu', 'ppiu']
        cid1 = [int(gdata.drug_idx_to_id[c][3:]) for c in self.drug1]
        cid2 = [int(gdata.drug_idx_to_id[c][3:]) for c in self.drug2]
        name1 = [gdata.drug_idx_to_name[c] for c in self.drug1]
        name2 = [gdata.drug_idx_to_name[c] for c in self.drug2]
        se_name = [gdata.side_effect_idx_to_name[c] for c in self.side_effect]

        if not self.if_pred:
            print('WARING: The query is not predicted')
            keys = keys[:8]
            df = [self.drug1, cid1, name1, self.drug2, cid2, name2, self.side_effect, se_name]
        else:
            df = [self.drug1, cid1, name1, self.drug2, cid2, name2, self.side_effect, se_name, self.probability, self.piu_score, self.ppiu_score]

        df = pandas.DataFrame(df).T
        df.columns = keys

        return df

    def get_GOEnrich_table(self):
        if not self.if_enrich:
            print('ERROR: There is no enriched GO item')
            return

        return self.GOEnrich_table

    def to_pickle(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
    
    def get_subgraph(self, if_show=True, save_path=None):
        if not self.if_explain:
            print('ERROR: The query is not explained')
            return

        G, self.fig = visualize_graph(self.pp_index, self.pp_weight, self.pd_index, self.pd_weight, gdata.pp_index, self.drug1, self.drug2, save_path, size=(30, 30), protein_name_dict=gdata.prot_graph_dict, drug_name_dict=gdata.drug_graph_dict)

        if if_show:
            self.fig.show()
        
        return G, self.fig


def visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, pp_adj, d1, d2,
                    out_path,
                    protein_name_dict=None, drug_name_dict=None, hiden=True,
                    size=(40, 40)):
    """
    :param pp_idx: integer tensor of the shape (2, n_pp_edges)
    :param pp_weight: float tensor of the shape (1, n_pp_edges), values within (0,1)
    :param pd_idx: integer tensor of the shape (2, n_pd_edges)
    :param pd_weight: float tensor of the shape (1, n_pd_edges), values within (0,1)
    :param protein_name_dict: store elements {protein_index -> protein name}
    :param drug_name_dict: store elements {drug_index -> drug name}

    1. use different color for pp and pd edges
    2. annotate the weight of each edge near the edge (or annotate with the tranparentness of edges for each edge)
    3. annotate the name of each node near the node, if name_dict=None, then annotate with node's index
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
        t1, t2 = e[0]
        if protein_name_dict.get(t1):
            t1 = protein_name_dict[t1]
        if protein_name_dict.get(t2):
            t2 = protein_name_dict[t2]

        # t1, t2 = protein_name_dict[e[0][0]], protein_name_dict[e[0][1]]
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
                    if protein_name_dict.get(i):
                        i = protein_name_dict[i]
                    if protein_name_dict.get(j):
                        j = protein_name_dict[j]
                    G.add_edge(i, j, weights='0')
                    pp_link.append((i, j))
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
