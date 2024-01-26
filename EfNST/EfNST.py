# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:36:44 2024

@author: lenovo
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import Sequential, BatchNorm
import scipy.sparse as sp
from torch_sparse import SparseTensor
import networkx as nx
class graph:
    def __init__(self, 
                 data, 
                 rad_cutoff,
                 k,  
                 distType='euclidean',):
        super(graph, self).__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff
        self.num_cell = data.shape[0]
    def graph_computing(self):
        dist_list = ["euclidean", "cosine"]
        graphList = []
        if self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k+1)
            indices = ind[:, 1:]
            graphList = [(node_idx, indices[node_idx][j]) for node_idx in range(self.data.shape[0]) for j in range(indices.shape[1])]
        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity', include_self=False)
            A = A.toarray()
            graphList = [(node_idx, indices[j]) for node_idx in range(self.data.shape[0]) for j in np.where(A[node_idx] == 1)[0]]
        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius=self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList = [(node_idx, indices[node_idx][j]) for node_idx in range(indices.shape[0]) for j in range(indices[node_idx].shape[0]) if distances[node_idx][j] > 0]
        return graphList
    def List2Dict(self, graphList):
        graphdict = {}
        tdict = {}
        for end1, end2 in graphList:
            tdict[end1] = ""
            tdict[end2] = ""
            graphdict.setdefault(end1, []).append(end2)
        for i in range(self.num_cell):
            if i not in tdict:
                graphdict[i] = []
        return graphdict
    def mx2SparseTensor(self, mx):
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).to(torch.long)
        col = torch.from_numpy(mx.col).to(torch.long)
        values = torch.from_numpy(mx.data)
        adj = SparseTensor(row=row, col=col, value=values, sparse_sizes=mx.shape)
        adj_ = adj.t()
        return adj_
    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self.mx2SparseTensor(adj_normalized)
    def main(self):
        adj_mtx = self.graph_computing()
        graph_dict = self.List2Dict(adj_mtx)
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
        adj_pre = adj_org - sp.dia_matrix((adj_org.diagonal()[np.newaxis, :], [0]), shape=adj_org.shape)
        adj_pre.eliminate_zeros()
        adj_norm = self.preprocess_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)
        graph_dict = {
            "adj_norm": adj_norm,
            "adj_label": adj_label,
            "norm_value": norm
        }
        return graph_dict
    def combine_graph_dicts(self, dict_1, dict_2):
        tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
        graph_dict = {
            "adj_norm": SparseTensor.from_dense(tmp_adj_norm),
            "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
            "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
        }
        return graph_dict
    
   
class EFNST_model(nn.Module):
    def __init__(self, input_dim,Conv_type='ResGatedGraphConv',linear_encoder_hidden=[50, 20],
                linear_decoder_hidden=[50, 60],conv_hidden=[32, 8],p_drop=0.1,
                dec_cluster_n=15,activate="relu"):
        super(EFNST_model, self).__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = 0.8
        self.conv_hidden = conv_hidden
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n
        current_encoder_dim = self.input_dim
        self.encoder = nn.Sequential()
        for le in range(len(linear_encoder_hidden)):
            self.encoder.add_module(f'encoder_L{le}', 
                                    buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate, self.p_drop))
            current_encoder_dim=linear_encoder_hidden[le]
        current_decoder_dim = linear_encoder_hidden[-1] + conv_hidden[-1]
        
        self.decoder = nn.Sequential()
        for ld in range(len(linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate, self.p_drop))
            current_decoder_dim= self.linear_decoder_hidden[ld]
        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}',buildNetwork(self.linear_decoder_hidden[-1], 
                                self.input_dim, "sigmoid", p_drop))
        if self.Conv_type == "ResGatedGraphConv":
            from torch_geometric.nn import ResGatedGraphConv
            self.conv = Sequential('x, edge_index', [
                        (ResGatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
        self.dc = InnerProductDecoder(p_drop)
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.linear_encoder_hidden[-1]+self.conv_hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def encode(self, x, adj):
        feat_x = self.encoder(x)
        conv_x = self.conv(feat_x, adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def target_distribution(self, target):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
    def EfNST_loss(self, decoded, x, preds, labels, mu, logvar, n_nodes, norm, mask=None, MSE_WT=10, KLD_WT=0.1):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)

        if mask is not None:
            preds = preds * mask
            labels = labels * mask
        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return MSE_WT * mse_loss + KLD_WT* (bce_logits_loss + KLD)
    def forward(self, x, adj):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat([feat_x, gnn_z], dim=1)
        de_feat = self.decoder(z)
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return z, mu, logvar, de_feat, q, feat_x, gnn_z
def buildNetwork(in_features, out_features, activate="relu", p_drop=0.0):
    layers = [
    nn.Linear(in_features, out_features),
    nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
    ]
    if activate == "relu":
        layers.append(nn.ELU())
    elif activate == "sigmoid":
        layers.append(nn.Sigmoid())
    if p_drop > 0:
        layers.append(nn.Dropout(p_drop))
    return nn.Sequential(*layers)
class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class GradientReverseLayer(torch.autograd.Function):
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x) * 1.0
    def backward(ctx, grad_output):
        return (grad_output * -1 * ctx.weight), None

class AdversarialNetwork(nn.Module):
    def __init__(self, model, n_domains: int = 2, weight: float = 1, n_layers: int = 2,)-> None:
        super(AdversarialNetwork, self).__init__()
        self.model = model
        self.n_domains = n_domains
        self.n_layers = n_layers
        self.weight = weight

        hidden_layers = [
            nn.Linear(self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1],
                      self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1]),
            nn.ReLU(),
        ] * n_layers

        self.domain_clf = nn.Sequential(
            *hidden_layers,
            nn.Linear(self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1], self.n_domains),
        )
        return
    def set_rev_grad_weight(self, weight: float) -> None:
        self.weight = weight
        return

    def target_distribution(self, target):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def EfNST_loss(
        self, decoded, x, preds, labels, mu, logvar, n_nodes, norm, mask=None, MSE_WT=10, KLD_WT=0.1
    ):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)

        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)

        KLD = -0.5 / n_nodes * torch.mean(
            torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1)
        )

        return MSE_WT * mse_loss + bce_logits_loss + KLD_WT * KLD

    def forward(self, x: torch.FloatTensor, edge_index) -> torch.FloatTensor:
        z, mu, logvar, de_feat, q, feat_x, gnn_z = self.model(x, edge_index)
        x_rev = GradientReverseLayer.apply(z, self.weight)
        domain_pred = self.domain_clf(x_rev)
        return z, mu, logvar, de_feat, q, feat_x, gnn_z, domain_pred
