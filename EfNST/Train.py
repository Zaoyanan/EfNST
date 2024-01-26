# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:10:01 2024

@author: lenovo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
from torch.autograd import Variable
import igraph as ig
import leidenalg
from sklearn.cluster import KMeans

class TrainingConfig:
    def __init__(self, pro_data, G_dict, model, pre_epochs, epochs,
                 corrupt=0.001, lr=5e-4, weight_decay=1e-4, domains=None,
                 KL_WT=100, MSE_WT=10, KLD_WT=0.1,
                 Domain_WT=1, use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.pro_data = pro_data
        self.data = torch.FloatTensor(pro_data.copy()).to(self.device)
        self.adj = G_dict['adj_norm'].to(self.device)
        self.adj_label = G_dict['adj_label'].to(self.device)
        self.norm = G_dict['norm_value']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=lr, weight_decay=weight_decay)
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.num_spots = self.data.shape[0]
        self.dec_tol = 0
        self.KL_WT = KL_WT
        self.q_stride = 20
        self.MSE_WT = MSE_WT
        self.KLD_WT = KLD_WT
        self.Domain_WT = Domain_WT
        self.corrupt = corrupt
        self.domains = torch.from_numpy(domains).to(self.device) if domains is not None else domains
    
    def masking_noise(data, frac):
        data_noise = data.clone()
        rand = torch.rand(data.size())
        data_noise[rand<frac] = 0
        return data_noise   
    def pretrain(self, grad_down=5):
        for epoch in range(self.pre_epochs):
            inputs_corr = TrainingConfig.masking_noise(self.data, self.corrupt)
            inputs_coor = inputs_corr.to(self.device)
            self.model.train()
            self.optimizer.zero_grad()            
            if self.domains is not None:
                z, mu, logvar, de_feat, _, feat_x, gnn_z, domain_pred = self.model(Variable(inputs_coor), self.adj)
                preds = self.model.model.dc(z)
            else:
                z, mu, logvar, de_feat, _, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
                preds = self.model.dc(z)
            loss = self.model.EfNST_loss(
                decoded=de_feat, 
                x=self.data, 
                preds=preds, 
                labels=self.adj_label, 
                mu=mu, 
                logvar=logvar, 
                n_nodes=self.num_spots, 
                norm=self.norm, 
                mask=self.adj_label, 
                MSE_WT=self.MSE_WT, 
                KLD_WT=self.KLD_WT,
            )
            if self.domains is not None:
                loss_function = nn.CrossEntropyLoss()
                Domain_loss = loss_function(domain_pred, self.domains)
                loss += Domain_loss * self.Domain_WT
            else:
                loss=loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_down)
            self.optimizer.step()
    def process(self):
        self.model.eval()
        if self.domains is None:
            z, _, _, _, q, _, _ = self.model(self.data, self.adj)
        else:
            z, _, _, _, q, _, _, _ = self.model(self.data, self.adj)
        z = z.cpu().detach().numpy()
        q = q.cpu().detach().numpy()        
        return z, q
    def save_and_load_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
    def fit(self, 
        cluster_n=20, 
        clusterType='leiden', 
        leiden_resolution=1.0,  
        pretrain=True,
        ):
        if pretrain:
            self.pretrain()
            pre_z, _ = self.process() 
        if clusterType == 'KMeans' and cluster_n is not None:  # 使用K均值算法进行聚类，且聚类数目已知
            cluster_method = KMeans(n_clusters=cluster_n, n_init=cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)    
        elif clusterType == 'Leiden':  
            if cluster_n is None:
                g = ig.Graph()
                g.add_vertices(pre_z.shape[0])
            for i in range(pre_z.shape[0]):
                for j in range(i+1, pre_z.shape[0]):
                    g.add_edge(i, j, weight=np.linalg.norm(pre_z[i] - pre_z[j]))            
            partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, resolution_parameter=leiden_resolution)
            y_pred_last = np.array(partition.membership)            
            unique_clusters = np.unique(y_pred_last)
            cluster_centers_ = np.array([pre_z[y_pred_last == cluster].mean(axis=0) for cluster in unique_clusters])
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
        else:
            cluster_method = KMeans(n_clusters=cluster_n, n_init=cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
    def train_epoch(self, epoch):
        self.model.train()
        if epoch % self.q_stride == 0:
            _, q = self.process()
            q = self.target_distribution(torch.Tensor(q).clone().detach())
            y_pred = q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
            self.y_pred_last = np.copy(y_pred)
            if epoch > 0 and delta_label < self.dec_tol:
                return False  
        torch.set_grad_enabled(True)
        self.optimizer.zero_grad()
        inputs_coor = self.data.to(self.device)
        if self.domains is None:
            z, mu, logvar, de_feat, out_q, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
            preds = self.model.dc(z)
        else:
            z, mu, logvar, de_feat, out_q, feat_x, gnn_z, domain_pred = self.model(Variable(inputs_coor), self.adj)
            loss_function = nn.CrossEntropyLoss()
            Domain_loss = loss_function(domain_pred, self.domains)
            preds = self.model.model.dc(z)
            loss_EfNST = self.model.EfNST_loss(
                decoded=de_feat,
                x=self.data,
                preds=preds,
                labels=self.adj_label,
                mu=mu,
                logvar=logvar,
                n_nodes=self.num_spots,
                norm=self.norm,
                mask=self.adj_label,
                MSE_WT=self.MSE_WT,
                KLD_WT=self.KLD_WT
                )
            loss_KL = F.KL_div(out_q.log(), q.to(self.device))
            if self.domains is None:
                loss = self.KL_WT * loss_KL + loss_EfNST
            else:
                loss = self.KL_WT * loss_KL + loss_EfNST + Domain_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
