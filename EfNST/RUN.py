# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:27:06 2024

@author: lenovo
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy.spatial import distance
from Defining import read_Visium,Refiner
from Image import Image_Feature, image_crop
from EfNST import graph, EFNST_model,AdversarialNetwork
from sklearn.metrics import calinski_harabasz_score
from Train import TrainingConfig
from AUG import augment_adata
class run():
	def __init__(self,save_path="./",pre_epochs=1000, epochs=500,pca_n_comps = 200,
              linear_encoder_hidden=[32,20],linear_decoder_hidden=[32],conv_hidden=[32,8],
              verbose=True,platform='Visium',cnnType='efficientnet-b0',Conv_type='ResGatedGraphConv',
              p_drop=0.01,dec_cluster_n=20,n_neighbors=15,min_cells=3,grad_down = 5,KL_WT = 100,
              MSE_WT = 10,KLD_WT = 0.1,Domain_WT = 1,use_gpu = True,
			):
		self.save_path = save_path
		self.pre_epochs = pre_epochs
		self.epochs = epochs
		self.pca_n_comps = pca_n_comps
		self.linear_encoder_hidden = linear_encoder_hidden
		self.linear_decoder_hidden = linear_decoder_hidden
		self.conv_hidden = conv_hidden
		self.verbose = verbose
		self.platform = platform
		self.cnnType = cnnType
		self.Conv_type = Conv_type
		self.p_drop = p_drop
		self.dec_cluster_n = dec_cluster_n
		self.n_neighbors = n_neighbors
		self.min_cells = min_cells
		self.platform = platform
		self.grad_down = grad_down
		self.KL_WT = KL_WT
		self.MSE_WT = MSE_WT
		self.KLD_WT = KLD_WT
		self.Domain_WT = Domain_WT
		self.use_gpu = use_gpu
	def _get_adata(
		self,
		data_path,
		data_name,
		verbose=True,
		):
		if self.platform =='Visium':
			adata = read_Visium(os.path.join(data_path, data_name))
		save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', f'{data_name}'))
		save_path_image_crop.mkdir(parents=True, exist_ok=True)
		adata = image_crop(adata, save_path=save_path_image_crop)
		adata = Image_Feature(adata, pca_components=self.pca_n_comps, cnnType=self.cnnType).Extract_Image_Feature()
		if verbose:
  			save_data_path = Path(os.path.join(self.save_path, f'{data_name}'))
  			save_data_path.mkdir(parents=True, exist_ok=True)
  			adata.write(os.path.join(save_data_path, f'{data_name}.h5ad'), compression="gzip")
		return adata
	def _get_graph(self,data,distType = "Radius",k = 12,rad_cutoff = 150,):
		graph_dict = graph(data, distType=distType, k=k, rad_cutoff=rad_cutoff).main()
		print("Step 2: Graph computing!")
		return graph_dict 
	def _get_augment(self,adata,Adj_WT = 0.2,neighbour_k = 4,weights = "weights_matrix_all",spatial_k = 30,):
		adata_augment = augment_adata(adata, 
								Adj_WT = Adj_WT,
								neighbour_k = neighbour_k,
								platform = self.platform,
								weights = weights,
								spatial_k = spatial_k,
								)  
		print("Step 1: Augment Gene!")
		return adata_augment
	def _optimize_cluster(
		self,
		adata,
		resolution_range=(0.1, 2.5, 0.01),
		):
		resolutions = np.arange(*resolution_range)
		scores = [
		calinski_harabasz_score(adata.X, sc.tl.leiden(adata, resolution=r).obs["leiden"])
                for r in resolutions
                ]
		cl_opt_df = pd.DataFrame({"resolution": resolutions, "score": scores})    
		best_resolution = cl_opt_df.loc[cl_opt_df["score"].idxmax(), "resolution"]    
		return best_resolution
	def _priori_cluster(self,adata,n_domains=7):
		for res in sorted(list(np.arange(0.1, 2.5, 0.01)), reverse=True):
			sc.tl.leiden(adata, random_state=0, resolution=res)
			count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
			if count_unique_leiden == n_domains:
				break
		return res   
	def _get__dataset_adata(self,data_path,data_name,character="spatial",
                         verbose=False,Adj_WT=0.2,neighbour_k=4,
                         weights="weights_matrix_all",spatial_k=30,
                         distType="Radius",k=12,rad_cutoff=150,):
		adata = self._get_adata(data_path=data_path, data_name=data_name, verbose=verbose)
		adata = self._get_augment(adata, Adj_WT=Adj_WT, 
								neighbour_k=neighbour_k, weights=weights, spatial_k=spatial_k)
		graph_dict = self._get_graph(adata.obsm[character], distType=distType, k=k,
									 rad_cutoff=rad_cutoff)
		self.data_name = data_name
		if self.verbose:
			print("Step 1: Augment Gene !")
			print("Step 2: Graph computing !")
		return adata,graph_dict
	def _fit(self,adata,graph_dict,domains=None,dim_reduction=True,pretrain=True,save_data=False,):
		print("Task sucessful, please wait")
		if self.platform == "Visium":
			adata.X = adata.obsm["augment_gene_data"].astype(float)
			if dim_reduction:
				sc.pp.filter_genes(adata, min_cells=self.min_cells)
				adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
				adata_X = sc.pp.log1p(adata_X)
				adata_X = sc.pp.scale(adata_X)
				concat_X = sc.pp.pca(adata_X, n_comps=self.pca_n_comps)
			else:
				sc.pp.filter_genes(adata, min_cells=self.min_cells)
				sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
				sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)
				sc.pp.log1p(adata)
				concat_X = adata[:, adata.var['highly_variable']].X
		else:
			concat_X = adata.obsm["augment_gene_data"]
		EfNST_model = EFNST_model(
							input_dim=concat_X.shape[1],
                        	Conv_type=self.Conv_type,
							linear_encoder_hidden=self.linear_encoder_hidden,
							linear_decoder_hidden=self.linear_decoder_hidden,
							conv_hidden=self.conv_hidden,
							p_drop=self.p_drop,
							dec_cluster_n=self.dec_cluster_n,
							)
		if domains is None:
			EfNST_training = TrainingConfig(concat_X,graph_dict,EfNST_model,
                                   pre_epochs=self.pre_epochs,epochs=self.epochs,
                                   KL_WT=self.KL_WT,MSE_WT=self.MSE_WT,
                                   KLD_WT=self.KLD_WT,Domain_WT=self.Domain_WT,
                                   use_gpu=self.use_gpu,)
		if pretrain:
			EfNST_training.fit()
		else:
			EfNST_training.pretrain(grad_down=self.grad_down)
		EfNST_embedding, _ = EfNST_training.process()
		if self.verbose:
			print("Step 3: Training Done!")
		adata.obsm["EfNST_embedding"] = EfNST_embedding
		return adata
	def _get_cluster_data(
   		self,
   		adata,
   		n_domains,
   		priori = True,
   		):
   		sc.pp.neighbors(adata, use_rep='EfNST_embedding', n_neighbors = self.n_neighbors)
   		if priori:
   			res = self._priori_cluster(adata, n_domains=n_domains)
   		else:
   			res = self._optimize_cluster(adata)
   		sc.tl.leiden(adata, key_added="EfNST_domain", resolution=res)
   		adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
   		refined_pred= Refiner.refine(sample_id=adata.obs.index.tolist(), 
   							 pred=adata.obs["EfNST_domain"].tolist(), dis=adj_2d, shape="hexagon")
   		adata.obs["EfNST"]= refined_pred
   		return adata





    
