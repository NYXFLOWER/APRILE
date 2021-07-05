import torch
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import numpy as np
import pandas
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
import os
import pickle

from goatools.base import \
    download_go_basic_obo  # Get http://geneontology.org/ontology/go-basic.obo
from goatools.base import \
    download_ncbi_associations  # Get ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
from goatools.test_data.genes_NCBI_9606_ProteinCoding import GENEID2NT as GeneID2nt_hum
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.godag_plot import plot_gos, plot_results, plot_goid2goobj

from aprile.utils import remove_bidirection, visualize_graph

# torch.manual_seed(1111)
# np.random.seed(1111)
# EPS = 1e-13

# # load data
# aprile_dir = os.path.dirname(os.path.abspath(__file__))
# with open(os.path.join(aprile_dir, 'data.pkl'), 'rb') as f:
#     gdata = pickle.load(f)


class MultiInnerProductDecoder(torch.nn.Module):
    """DistMult tensor factorization for side effect prediction, 

    Args:
        in_dim (int): the dimension of drug feature
        num_et (int): the number of side effect
    """
    def __init__(self, in_dim, num_et):
        super(MultiInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Parameter(torch.Tensor(num_et, in_dim))

        self._reset_parameters()

    def forward(self, z, edge_index, edge_type, sigmoid=True):
        """forward propagation to predict {(drug, drug, side_effect)} 

        Args:
            z (torch.Tensor): drug features
            edge_index (torch.Tensor): sparse representation of DDIs
            edge_type (torch Tensor): side effect associated with each edge index
            sigmoid (bool, optional): if apply Sigmoid. Defaults to True.

        Returns:
            torch tensor: probability of DDIs with associated 
        """
        value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def _reset_parameters(self):
        self.weight.data.normal_(std=1/np.sqrt(self.in_dim))


# class AprileGCN(MessagePassing):
#     """Graph convolutional neural network [1] with edge weights/masks.
#         [1]: `"Semi-supervised Classification with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>` (ICLR 2017) paper.

#     Note: 
#         For more information please see Pytorch Geomertic's `nn.GCNConv <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.conv.message_passing>`_ docs.

#     Args:
#         in_channels (int): size of each inputs samples
#         out_channels ([type]): size of each outputs samples
#         improved (bool, optional): Defaults to False.
#         cached (bool, optional): Defaults to False.
#         bias (bool, optional): Defaults to True.
#     """
#     def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=True, **kwargs):
#         super(AprileGCN, self).__init__(aggr='add', **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.improved = improved
#         self.cached = cached
#         self.cached_result = None

#         self.weight = Parameter(torch.Tensor(in_channels, out_channels))

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = np.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
#         self.weight.data.uniform_(-stdv, stdv)

#         if self.bias is not None:
#             self.bias.data.fill_(0)

#         self.cached_result = None
#         self.cached_num_edges = None

#     @staticmethod
#     def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
#         """Add self-loops and apply symmetric normalization
#         """
#         if edge_weight is None:
#             edge_weight = torch.ones((edge_index.size(1), ),
#                                      dtype=dtype,
#                                      device=edge_index.device)

#         fill_value = 1 if not improved else 2
#         edge_index, edge_weight = add_remaining_self_loops(
#             edge_index, edge_weight, fill_value, num_nodes)

#         row, col = edge_index
#         deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

#         return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

#     def forward(self, x, edge_index, edge_weight=None):
#         x = torch.matmul(x, self.weight)

#         if self.cached and self.cached_result is not None:
#             if edge_index.size(1) != self.cached_num_edges:
#                 raise RuntimeError(
#                     'Cached {} number of edges, but found {}'.format(
#                         self.cached_num_edges, edge_index.size(1)))

#         if not self.cached or self.cached_result is None:
#             self.cached_num_edges = edge_index.size(1)
#             edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
#                                          self.improved, x.dtype)
#             self.cached_result = edge_index, norm

#         edge_index, norm = self.cached_result

#         return self.propagate(edge_index, x=x, norm=norm)

#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j

#     def update(self, aggr_out):
#         if self.bias is not None:
#             aggr_out = aggr_out + self.bias
#         return aggr_out

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
#                                    self.out_channels)


# class PP(torch.nn.Module):
#     """Protein representation module

#     Args:
#         in_dim (int): the size of each input samples
#         nhid_list (list): the size of each intermediary embeddings and outputs
#     """

#     def __init__(self, in_dim, nhid_list):
#         super(PP, self).__init__()
#         self.out_dim = nhid_list[-1]

#         self.embedding = torch.nn.Parameter(torch.Tensor(in_dim, nhid_list[0]))

#         self.conv_list = torch.nn.ModuleList(
#             [AprileGCN(nhid_list[i], nhid_list[i + 1], cached=True) for i in range(len(nhid_list) - 1)]
#         )

#         self.embedding.requires_grad = False
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.embedding.data.normal_()

#     def forward(self, x, pp_edge_index, edge_weight):
#         tmp = []

#         x = self.embedding
#         tmp.append(x)
#         for net in self.conv_list[:-1]:
#             x = net(x, pp_edge_index, edge_weight)
#             x = F.relu(x, inplace=True)
#             tmp.append(x)

#         x = self.conv_list[-1](x, pp_edge_index, edge_weight)
#         x = F.relu(x, inplace=True)
#         tmp.append(x)

#         return torch.cat(tmp, dim=1)


# class PD(torch.nn.Module):
#     """Drug representation module

#     Args:
#         protein_dim (int): the size of protein embeddings
#         d_dim_prot (int): the size of drug embeddings for the related pharmacogenomic information
#         n_drug (int): the number of drugs
#         d_dim_feat (int, optional): the size of drug feature embeddings. Defaults to 32.
#     """

#     def __init__(self, protein_dim, d_dim_prot, n_drug, d_dim_feat=32):
#         super(PD, self).__init__()
#         self.p_dim = protein_dim
#         self.d_dim_prot = d_dim_prot
#         self.d_dim_feat = d_dim_feat
#         self.n_drug = n_drug
#         self.d_feat = torch.nn.Parameter(torch.Tensor(n_drug, d_dim_feat))

#         self.d_feat.requires_grad = False

#         self.conv = AprileGCN(protein_dim, d_dim_prot, cached=True)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.d_feat.data.normal_()

#     def forward(self, x, pd_edge_index, edge_weight=None):
#         n_prot = x.shape[0]
#         tmp = pd_edge_index + 0
#         tmp[1, :] += n_prot

#         x = torch.cat([x, torch.zeros((self.n_drug, x.shape[1])).to(x.device)], dim=0)
#         x = self.conv(x, tmp, edge_weight)[n_prot:, :]
#         x = F.relu(x)
#         x = torch.cat([x, torch.abs(self.d_feat)], dim=1)
#         return x


# class AprilePredModel(torch.nn.Module):
#     """Aprile-Pred model structure
#     """
#     def __init__(self, pp, pd, mip):
#         super(AprilePredModel, self).__init__()
#         self.pp = pp
#         self.pd = pd
#         self.mip = mip


# class Pre_mask(torch.nn.Module):
#     """AprileGCN edge masker for AprileExplainer

#     Args:
#         pp_n_link (int): the number of protein-protein edges
#         pd_n_link (int): the number of protein-drug edges
#     """
#     def __init__(self, pp_n_link, pd_n_link):
#         super(Pre_mask, self).__init__()
#         self.pp_weight = Parameter(torch.Tensor(pp_n_link))
#         self.pd_weight = Parameter(torch.Tensor(pd_n_link))
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.pp_weight.data.fill_(0.99)
#         self.pd_weight.data.fill_(0.99)

#     # no change when at 0 or 1
#     def desaturate(self):
#         mask = self.pp_weight.data > 0.99
#         self.pp_weight.data[mask] = 0.99

#         mask = self.pp_weight.data < 0.01
#         self.pp_weight.data[mask] = 0.01

#         mask = self.pd_weight.data > 0.99
#         self.pd_weight.data[mask] = 0.99

#         mask = self.pd_weight.data < 0.01
#         self.pd_weight.data[mask] = 0.01

#     def saturate(self):
#         mask = self.pp_weight.data >= 0.99
#         self.pp_weight.data[mask] = 1.0

#         mask = self.pp_weight.data <= 0.01
#         self.pp_weight.data[mask] = 0.0

#         mask = self.pd_weight.data >= 0.99
#         self.pd_weight.data[mask] = 1.0

#         mask = self.pd_weight.data <= 0.01
#         self.pd_weight.data[mask] = 0.0


# class AprileExplainer(object):
#     """Explain APRILE-Predictor's predictions by given a small set of drug targets and protein-protein interactions
#     """
#     def __init__(self, model, data, device):
#         super(AprileExplainer, self).__init__()
#         self.model = model
#         self.data = data
#         self.device = device

#     def explain(self, drug_list_1, drug_list_2, side_effect_list, regularization=1):
#         data = self.data
#         model = self.model
#         device = self.device

#         pre_mask = Pre_mask(data.pp_index.shape[1] // 2, data.pd_index.shape[1]).to(device)
#         data = data.to(device)
#         model = model.to(device)

#         for gcn in self.model.pp.conv_list:
#             gcn.cached = False
#         self.model.pd.conv.cached = False
#         self.model.eval()

#         pp_static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)
#         pd_static_edge_weights = torch.ones((data.pd_index.shape[1])).to(device)

#         optimizer = torch.optim.Adam(pre_mask.parameters(), lr=0.01)
#         fake_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#         z = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)
#         z = model.pd(z, data.pd_index, pd_static_edge_weights)

#         P = torch.sigmoid((z[drug_list_1] * z[drug_list_2] * model.mip.weight[side_effect_list]).sum(dim=1))

#         if len(drug_list_1) < 5:
#             print(P.tolist())

#         tmp = 0.0
#         pre_mask.reset_parameters()
#         for i in range(9999):
#             model.train()
#             pre_mask.desaturate()
#             optimizer.zero_grad()
#             fake_optimizer.zero_grad()

#             half_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pp_weight)
#             pp_mask = torch.cat([half_mask, half_mask])

#             pd_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pd_weight)

#             z = model.pp(data.p_feat, data.pp_index, pp_mask)

#             z = model.pd(z, data.pd_index, pd_mask)

#             P = torch.sigmoid((z[drug_list_1] * z[drug_list_2] * model.mip.weight[side_effect_list]).sum(dim=1))
#             EPS = 1e-7

#             loss = torch.log(1 - P + EPS).sum() / regularization \
#                    + 0.5 * (pp_mask * (2 - pp_mask)).sum() \
#                    + (pd_mask * (2 - pd_mask)).sum()

#             loss.backward()
#             optimizer.step()
#             if i % 100 == 0:
#                 print("Epoch:{:3d}, loss:{:0.2f}, prob:{:0.2f}, pp_link_sum:{:0.2f}, pd_link_sum:{:0.2f}".format(i, loss.tolist(), P.mean().tolist(), pp_mask.sum().tolist(), pd_mask.sum().tolist()))

#             if tmp == (pp_mask.sum().tolist(), pd_mask.sum().tolist()):
#                 break
#             else:
#                 tmp = (pp_mask.sum().tolist(), pd_mask.sum().tolist())

#         pre_mask.saturate()

#         pp_left_mask = (pp_mask > 0.2).detach().cpu().numpy()
#         tmp = (data.pp_index[0, :] > data.pp_index[1, :]).detach().cpu().numpy()
#         pp_left_mask = np.logical_and(pp_left_mask, tmp)

#         pd_left_mask = (pd_mask > 0.2).detach().cpu().numpy()

#         pp_left_index = data.pp_index[:, pp_left_mask].cpu().numpy()
#         pd_left_index = data.pd_index[:, pd_left_mask].cpu().numpy()

#         pp_left_weight = pp_mask[pp_left_mask].detach().cpu().numpy()
#         pd_left_weight = pd_mask[pd_left_mask].detach().cpu().numpy()

#         return pp_left_index, pp_left_weight, pd_left_index, pd_left_weight


# class AprilePredictorPretrained(object):
#     """Make adverse polypharmacy reactions using a pre-trained predictor

#     Args:
#         data_path (str): the path of pre-trained Aprile-Predictor
#     """
#     def __init__(self, data_path):
#         # load data
#         with open(data_path, 'rb') as f:
#             self.data = pickle.load(f)

#         # load pretrained model
#         self.model, self.name = self.__pretrained_model_construction__()
#         self.model.load_state_dict(
#             torch.load(os.path.join(aprile_dir, self.name + '-model.pt')))

#     def __pretrained_model_construction__(self):
#         nhids_gcn = [64, 32, 32]
#         prot_out_dim = sum(nhids_gcn)
#         drug_dim = 128
#         pp = PP(self.data.n_prot, nhids_gcn)
#         pd = PD(prot_out_dim, drug_dim, self.data.n_drug)
#         mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, self.data.n_et)
#         name = 'poly-' + str(nhids_gcn) + '-' + str(drug_dim)

#         return AprilePredModel(pp, pd, mip).to('cpu'), name

#     def predict(self, drug1, drug2, side_effect, device='cpu', threshold=0.5):
#         data = self.data.to(device)
#         model = self.model.to(device)
#         model.eval()

#         pp_static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)
#         pd_static_edge_weights = torch.ones((data.pd_index.shape[1])).to(device)
#         z = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)
#         z = model.pd(z, data.pd_index, pd_static_edge_weights)
#         P = torch.sigmoid(
#             (z[drug1] * z[drug2] * model.mip.weight[side_effect]).sum(dim=1)
#         ).to('cpu')

#         index_filter = P > threshold
#         drug1 = torch.Tensor(drug1)[index_filter].tolist()
#         if not drug1:
#             raise ValueError("No Satisfied Edges."
#                              + "\n - Suggestion: reduce the threshold probability."
#                              + "Current probability threshold is {}. ".format(threshold)
#                              + "\n - Please use -h for help")

#         drug2 = torch.Tensor(drug2)[index_filter].tolist()
#         side_effect = torch.Tensor(side_effect)[index_filter].tolist()

#         return drug1, drug2, side_effect, P[index_filter].tolist()

# class Aprile(object):
#     """APRILE: explaning polypharmacy side effect using a pre-trained APRILE-Pred model

#     Args:
#         device (str): 'cpu' or 'cuda', for running APRILE-Explainer
#     """
#     def __init__(self, device='cpu'):
#         # load pretrained model
#         self.model, self.name = self.__pretrained_model_construction__()
#         self.model.load_state_dict(torch.load(os.path.join(aprile_dir, 'POSE-pred.pt')))

#         self.device = device
#         self.__GO_enrich__()


#     def __GO_enrich__(self):
#         """Gene Ontology (GO) enrichment analysis
#         """
#         go_file = "go-basic.obo"
#         if not os.path.exists(go_file):
#             download_go_basic_obo()
        
#         # Load gene ontologies
#         obodag = GODag("go-basic.obo")
   
#         # Read NCBI's gene2go. Store annotations in a list of namedtuples
#         fin_gene2go = download_ncbi_associations()
#         objanno = Gene2GoReader(fin_gene2go, taxids=[9606])
#         # Get namespace2association where:
#         #    namespace is:
#         #        BP: biological_process
#         #        MF: molecular_function
#         #        CC: cellular_component
#         #    association is a dict:
#         #        key: NCBI GeneID
#         #        value: A set of GO IDs associated with that gene
#         ns2assoc = objanno.get_ns2assc()

#         self.goeaobj = GOEnrichmentStudyNS(
#             GeneID2nt_hum.keys(),  # List of human protein-acoding genes
#             ns2assoc,  # geneID/GO associations
#             obodag,  # Ontologies
#             propagate_counts=False,
#             alpha=0.05,  # default significance cut-off
#             methods=['fdr_bh'])  # default multipletest correction method

#     def __pretrained_model_construction__(self):
#         """construct the default pretrained APRILE-pred model

#         Returns:
#             Model: initial the model structure
#             str: model structure summary
#         """
#         nhids_gcn = [64, 32, 32]
#         prot_out_dim = sum(nhids_gcn)
#         drug_dim = 128
#         pp = PP(gdata.n_prot, nhids_gcn)
#         pd = PD(prot_out_dim, drug_dim, gdata.n_drug)
#         mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, gdata.n_et)
#         name = 'poly-' + str(nhids_gcn) + '-' + str(drug_dim)

#         return AprilePredModel(pp, pd, mip).to('cpu'), name

#     def get_prediction_train(self, threshold=0.5):
#         """generate predictions for DDIs in the training set

#         Args:
#             threshold (float, optional): the threshold of probability scores for DDIs. Defaults to 0.5.

#         Returns:
#             AprileQuery: prediction results
#         """
#         train_idx, train_et = remove_bidirection(gdata.train_idx, gdata.train_et)

#         return self.predict(train_idx[0].tolist(), train_idx[1].tolist(), train_et.tolist(), threshold=threshold)

#     def get_prediction_test(self, threshold=0.5):
#         """generate predictions for DDIs in the testing set

#         Args:
#             threshold (float, optional): the threshold of probability scores for DDIs. Defaults to 0.5.

#         Returns:
#             AprileQuery: prediction results
#         """
#         test_idx, test_et = remove_bidirection(gdata.test_idx, gdata.test_et)

#         return self.predict(test_idx[0].tolist(), test_idx[1].tolist(), test_et.tolist(), threshold=threshold)

#     def predict(self, drug1, drug2, side_effect, threshold=0.5):
#         """Predict the probability of DDIs

#         Args:
#             drug1 (list): a list of drug
#             drug2 (list): a list of drug pairing `drug1`
#             side_effect (list): a list of side effect
#             threshold (float, optional): for probability. Defaults to 0.5.

#         Raises:
#             ValueError: None of DDIs meets the threshold

#         Returns:
#             AprileQuery: prediction results
#         """
#         device = self.device
#         data = gdata.to(device)
#         model = self.model.to(device)
#         model.eval()

#         pp_static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)
#         pd_static_edge_weights = torch.ones((data.pd_index.shape[1])).to(device)
#         z = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)
#         z0 = z.clone()
#         z1 = z.clone()

#         # prediction based on all info
#         z = model.pd(z, data.pd_index, pd_static_edge_weights)
#         P = torch.sigmoid(
#             (z[drug1] * z[drug2] * model.mip.weight[side_effect]).sum(dim=1)
#         ).to('cpu')

#         index_filter = P > threshold
#         drug1 = torch.Tensor(drug1)[index_filter].numpy().astype(int).tolist()
#         if not drug1:
#             raise ValueError("No Satisfied Edges."
#                              + "\n - Suggestion: reduce the threshold probability."
#                              + "Current probability threshold is {}. ".format(threshold)
#                              + "\n - Please use -h for help")

#         drug2 = torch.Tensor(drug2)[index_filter].numpy().astype(int).tolist()
#         side_effect = torch.Tensor(side_effect)[index_filter].numpy().astype(int).tolist()

#         # prediction based on protein info and their interactions
#         z0.data[:, 64:] *= 0
#         z0 = model.pd(z0, data.pd_index, pd_static_edge_weights)
#         P0 = torch.sigmoid((z0[drug1] * z0[drug2] * model.mip.weight[side_effect]).sum(dim=1)).to("cpu")
#         ppiu_score = (P[index_filter] - P0)/P[index_filter]

#         # prediction based on drug info only
#         z1.data *= 0
#         z1 = model.pd(z1, data.pd_index, pd_static_edge_weights)
#         P1 = torch.sigmoid((z1[drug1] * z1[drug2] * model.mip.weight[side_effect]).sum(dim=1)).to("cpu")
#         piu_score = (P[index_filter] - P1)/P[index_filter]

#         # return a query object
#         query = AprileQuery(drug1, drug2, side_effect)
#         query.set_pred_result(P[index_filter].tolist(), piu_score.tolist(), ppiu_score.tolist())

#         return query 

#     def explain_list(self, drug_list_1, drug_list_2, side_effect_list, regularization=2, if_auto_tuning=True, if_pred=True):
#         """generate explanation for a list of adverse drug events"""
#         if if_pred:
#             query = self.predict(drug_list_1, drug_list_2, side_effect_list)
#         else:    
#             query = AprileQuery(drug_list_1, drug_list_2, side_effect_list, regularization)
#         return self.explain_query(query, if_auto_tuning=if_auto_tuning, regularization=query.regularization)

#     def explain_query(self, query, if_auto_tuning=True, regularization=2):
#         """generate explanation for a AprileEQuery query"""
#         query.regularization = regularization

#         pp_left_index, pp_left_weight, pd_left_index, pd_left_weight = self.__explain(query)

#         if if_auto_tuning:
#             while pp_left_index.shape[1]==0:
#                 if query.regularization < 0.0001:
#                     print("Warning: auto tuning forced to stop.")
#                     break
#                 query.regularization /= 2
#                 pp_left_index, pp_left_weight, pd_left_index, pd_left_weight = self.__explain(query)

#         query.set_exp_result(pp_left_index, pp_left_weight, pd_left_index, pd_left_weight)

#         goea_results_sig = self.enrich_go(pp_left_index)
#         query.set_enrich_result(goea_results_sig)

#         return query

#     def enrich_go(self, pp_left_index):
#         """gene ontology enrichment analysis"""
#         geneids_study = pp_left_index.flatten()  # geneid2symbol.keys()
#         geneids_study = [int(gdata.prot_idx_to_id[idx].replace('GeneID', '')) for idx in geneids_study]

#         goea_results_all = self.goeaobj.run_study(geneids_study)
#         goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]
        
#         return goea_results_sig
        
#     def __explain(self, query):
#         """the AprileExplainer module"""
#         data = gdata
#         model = self.model
#         device = self.device
        
#         drug_list_1, drug_list_2, side_effect_list, regularization = query.get_query()

#         pre_mask = Pre_mask(data.pp_index.shape[1] // 2, data.pd_index.shape[1]).to(device)
#         data = data.to(device)
#         model = model.to(device)

#         for gcn in self.model.pp.conv_list:
#             gcn.cached = False
#         self.model.pd.conv.cached = False
#         self.model.eval()

#         optimizer = torch.optim.Adam(pre_mask.parameters(), lr=0.01)
#         fake_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#         tmp = 0.0
#         pre_mask.reset_parameters()
#         for i in range(9999):
#             model.train()
#             pre_mask.desaturate()
#             optimizer.zero_grad()
#             fake_optimizer.zero_grad()

#             half_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pp_weight)
#             pp_mask = torch.cat([half_mask, half_mask])

#             pd_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pd_weight)

#             z = model.pp(data.p_feat, data.pp_index, pp_mask)

#             z = model.pd(z, data.pd_index, pd_mask)

#             P = torch.sigmoid((z[drug_list_1] * z[drug_list_2] * model.mip.weight[side_effect_list]).sum(dim=1))
#             EPS = 1e-7

#             loss = torch.log(1 - P + EPS).sum() / regularization \
#                    + 0.5 * (pp_mask * (2 - pp_mask)).sum() \
#                    + (pd_mask * (2 - pd_mask)).sum()

#             loss.backward()
#             optimizer.step()
            
#             if i % 100 == 0:
#                 print("Epoch:{:3d}, loss:{:0.2f}, prob:{:0.2f}, pp_link_sum:{:0.2f}, pd_link_sum:{:0.2f}".format(i, loss.tolist(), P.mean().tolist(), pp_mask.sum().tolist(), pd_mask.sum().tolist()))

#             # until no weight need to be updated --> no sum of weights changes
#             if tmp == (pp_mask.sum().tolist(), pd_mask.sum().tolist()):
#                 break
#             else:
#                 tmp = (pp_mask.sum().tolist(), pd_mask.sum().tolist())


#         pre_mask.saturate()

#         pp_left_mask = (pp_mask > 0.2).detach().cpu().numpy()
#         tmp = (data.pp_index[0, :] > data.pp_index[1, :]).detach().cpu().numpy()
#         pp_left_mask = np.logical_and(pp_left_mask, tmp)

#         pd_left_mask = (pd_mask > 0.2).detach().cpu().numpy()

#         pp_left_index = data.pp_index[:, pp_left_mask].cpu().numpy()
#         pd_left_index = data.pd_index[:, pd_left_mask].cpu().numpy()

#         pp_left_weight = pp_mask[pp_left_mask].detach().cpu().numpy()
#         pd_left_weight = pd_mask[pd_left_mask].detach().cpu().numpy()

#         return pp_left_index, pp_left_weight, pd_left_index, pd_left_weight

     
# class AprileQuery(object):
#     """A class for quering APRILE's prediction and expalanition results

#     Args:
#         drug1 (list): a list of drug
#         drug2 (list): a list of drug pairing with `drug1`
#         side_effect (list): a list of side effect caused by drug pairs
#         regularization (int, optional): the coefficient for control the size of explanation. Defaults to 2.
#     """
#     def __init__(self, drug1, drug2, side_effect, regularization=2):
#         self.drug1 = drug1
#         self.drug2 = drug2
#         self.side_effect = side_effect
#         self.regularization = regularization
#         self.if_explain = False
#         self.if_enrich = False
#         self.if_pred = False

#     def __repr__(self):
#         return str(self.__class__) + ": \n" + str(self.__dict__)
        
#     def __str__(self):
#         return str(self.__class__) + ": " + str(self.__dict__)

#     def set_exp_result(self, pp_index, pp_weight, pd_index, pd_weight):
#         if pd_index.shape[1]:
#             self.pp_index = pp_index
#             self.pp_weight = pp_weight
#             self.pd_index = pd_index
#             self.pd_weight = pd_weight
#             self.if_explain = True

#             print('pp_edge: {}, pd_edge:{}\n'.format(pp_index.shape[1], pd_index.shape[1]))

#     def set_enrich_result(self, goea_results_sig):
#         if len(goea_results_sig):
#             self.if_enrich = True

#             keys = ['name', 'namespace', 'id']
#             df_go1 = pandas.DataFrame([{k: g.goterm.__dict__.get(k) for k in keys} for g in goea_results_sig])
#             df_p = pandas.DataFrame([{'p_fdr_bh': g.__dict__['p_fdr_bh']} for g in goea_results_sig])
#             df_go = df_go1.merge(df_p, left_index=True, right_index=True)

#             go_genes = pandas.DataFrame([{'id': g.goterm.id, 'gene': s, 'symbol': gdata.geneid2symbol[s]} for g in goea_results_sig for s in g.study_items])
        
#             self.GOEnrich_table = df_go.merge(go_genes, on='id')

#     def set_pred_result(self, probability, piu_score, ppiu_score):
#         self.probability = probability
#         self.piu_score = piu_score
#         self.ppiu_score = ppiu_score
#         self.if_pred = True

#     def get_query(self):
#         """get query details
#         """
#         return self.drug1, self.drug2, self.side_effect, self.regularization
        
#     def get_pred_table(self):
#         """generate the prediction results

#         Returns:
#             pandas.DataFrame: DDIs, probability, PIU, PPIU and additional mappings
#         """
#         keys = ['drug_1', 'CID_1', 'name_1', 'drug_2', 'CID_2', 'name_2', 'side_effect', 'side_effect_name', 'prob', 'piu', 'ppiu']
#         cid1 = [int(gdata.drug_idx_to_id[c][3:]) for c in self.drug1]
#         cid2 = [int(gdata.drug_idx_to_id[c][3:]) for c in self.drug2]
#         name1 = [gdata.drug_idx_to_name[c] for c in self.drug1]
#         name2 = [gdata.drug_idx_to_name[c] for c in self.drug2]
#         se_name = [gdata.side_effect_idx_to_name[c] for c in self.side_effect]

#         if not self.if_pred:
#             print('WARING: The query is not predicted')
#             keys = keys[:8]
#             df = [self.drug1, cid1, name1, self.drug2, cid2, name2, self.side_effect, se_name]
#         else:
#             df = [self.drug1, cid1, name1, self.drug2, cid2, name2, self.side_effect, se_name, self.probability, self.piu_score, self.ppiu_score]

#         df = pandas.DataFrame(df).T
#         df.columns = keys

#         return df

#     def get_GOEnrich_table(self):
#         """get Gene Ontology analysis results

#         Returns:
#             pandas.DataFrame: significant GOs, genes and additional mappings
#         """
#         if not self.if_enrich:
#             print('ERROR: There is no enriched GO item')
#             return

#         return self.GOEnrich_table
    
#     def get_subgraph(self, if_show=True, save_path=None):
#         """Visualize explanation

#         Args:
#             if_show (bool, optional): if print the figure. Defaults to True.
#             save_path (str, optional): the path to save figure. Defaults to None.

#         Returns:
#             matplotlib.pyplot.figure(): DDIs and their mechanisms
#         """
#         if not self.if_explain:
#             print('ERROR: The query is not explained')
#             return

#         _, self.fig = visualize_graph(self.pp_index, self.pp_weight, self.pd_index, self.pd_weight, gdata.pp_index, self.drug1, self.drug2, save_path, size=(30, 30), protein_name_dict=gdata.prot_graph_dict, drug_name_dict=gdata.drug_graph_dict)

#         if if_show:
#             self.fig.show()
        
#         return self.fig

#     @staticmethod
#     def load_from_pkl(file):
#         """load a query from a pickle file

#         Args:
#             file (str): the file's path

#         Returns:
#             AprileQuery: 
#         """
#         with open(file, 'rb') as f:
#             return pickle.load(f)

#     def to_pickle(self, file):
#         """save the current query to a pickle file

#         Args:
#             file (str): the path to save the object
#         """
#         with open(file, 'wb') as f:
#             pickle.dump(self, f)
