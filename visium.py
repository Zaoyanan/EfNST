# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:58:58 2024

@author: lenovo
"""

import matplotlib.pyplot as plt
import scanpy as sc
from RUN import run
data_path = "C:/Users/lenovo/Desktop"
#data_path = "D:/科研/ST数据/DLPFC12切片"
save_path = "C:/Users/lenovo/Desktop/1515" #### save path
#data_name="V1_Breast_Cancer_Block_A_Section_1"
data_name="151510"
quality='hires'
n_domains=5
EfNST= run(save_path = save_path,platform = "Visium",pca_n_comps = 200,pre_epochs = 800, #### According to your own hardware, choose the number of training
           epochs = 1000,Conv_type="ResGatedGraphConv")
adata= EfNST._get_adata(data_path, data_name)
adata = EfNST._get_augment(adata,  neighbour_k = 4,)
graph_dict = EfNST._get_graph(adata.obsm["spatial"], distType="KDTree", k=12)
adata = EfNST._fit(adata, graph_dict, pretrain = False)
adata= EfNST._get_cluster_data(adata, n_domains = n_domains, priori=True)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color=["EfNST"],
              title='EfNST',show=False,)
