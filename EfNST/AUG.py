# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:54:08 2024

@author: lenovo
"""
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix

import numpy as np
from sklearn.linear_model import LinearRegression
import math
import pandas as pd
#class Aug:
    # def __init__(self, data,spatial_k=50, spatial_type="KDTree"):
    #     if spatial_type != "KDTree":
    #         raise ValueError("Invalid spatial_type. Supported types are: 'KDTree'.")       
    #     self.data = data
    #     self.spatial_k = spatial_k
    #     self.spatial_type = spatial_type
# def cal_spatial_weight(data,spatial_k = 50,spatial_type = "KDTree",):
# 	from sklearn.neighbors import KDTree
# 	if spatial_type == "KDTree":
# 		tree = KDTree(data, leaf_size=2)
# 		_, indices = tree.query(data, k=spatial_k + 1)
# 	indices = indices[:, 1:]
# 	spatial_weight = np.zeros((data.shape[0], data.shape[0]))
# 	for i in range(indices.shape[0]):
# 		ind = indices[i]
# 		for j in ind:
# 			spatial_weight[i][j] = 1
# 	return spatial_weight
def cal_spatial_weight(
	data,
	spatial_k = 50,
	spatial_type = "KDTree",
	):
	from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
	if spatial_type == "NearestNeighbors":
		nbrs = NearestNeighbors(n_neighbors=spatial_k+1, algorithm='ball_tree').fit(data)
		_, indices = nbrs.kneighbors(data)
	elif spatial_type == "KDTree":
		tree = KDTree(data, leaf_size=2) 
		_, indices = tree.query(data, k=spatial_k+1)
	elif spatial_type == "BallTree":
		tree = BallTree(data, leaf_size=2)
		_, indices = tree.query(data, k=spatial_k+1)
	indices = indices[:, 1:]
	spatial_weight = np.zeros((data.shape[0], data.shape[0]))
	for i in range(indices.shape[0]):
		ind = indices[i]
		for j in ind:
			spatial_weight[i][j] = 1
	return spatial_weight
def cal_gene_weight(data, n_components=50, gene_dist_type="cosine"):

    pca = PCA(n_components = n_components)
    if isinstance(data, np.ndarray):
        data_pca = pca.fit_transform(data)
    elif isinstance(data, csr_matrix):
        data = data.toarray()
        data_pca = pca.fit_transform(data)
    gene_correlation = 1 - pairwise_distances(data_pca, metric = gene_dist_type)
    return gene_correlation
def cal_weight_matrix(adata,platform="Visium", pd_dist_type="euclidean", md_dist_type="cosine", gb_dist_type="correlation",
                      n_components=50, no_morphological=True, spatial_k=30, spatial_type="KDTree", verbose=False):
	if platform == "Visium":
		img_row = adata.obs["imagerow"]
		img_col = adata.obs["imagecol"]
		array_row = adata.obs["array_row"]
		array_col = adata.obs["array_col"]
		rate = 3
		reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
		reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
		physical_distance = pairwise_distances(
            adata.obs[["imagecol", "imagerow"]],metric=pd_dist_type)
		unit = math.sqrt(reg_row.coef_ ** 2 + reg_col.coef_ ** 2)
		physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
	else:
		physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type)
    
	gene_counts = adata.X.copy()
	gene_correlation = cal_gene_weight(data=gene_counts, gene_dist_type=gb_dist_type, n_components=n_components)
	del gene_counts
	if verbose:
		adata.obsm["gene_correlation"] = gene_correlation
		adata.obsm["physical_distance"] = physical_distance

	if platform == 'Visium':
		morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric=md_dist_type)
		morphological_similarity[morphological_similarity < 0] = 0
		if verbose:
			adata.obsm["morphological_similarity"] = morphological_similarity
		adata.obsm["weights_matrix_all"] = (physical_distance
                              * gene_correlation
                              * morphological_similarity)
		if no_morphological:
			adata.obsm["weights_matrix_nomd"] = (gene_correlation
                               * physical_distance)
	else:
		adata.obsm["weights_matrix_nomd"] = (gene_correlation
                               * physical_distance)
	return adata

def find_adjacent_spot(adata, use_data="raw", neighbour_k=4, weights='weights_matrix_all', verbose=False):
	if use_data == "raw":
		if isinstance(adata.X, (csr_matrix, np.ndarray)):
			gene_matrix = adata.X.toarray()                
		elif isinstance(adata.X, np.ndarray):
			gene_matrix = adata.X
		elif isinstance(adata.X, pd.Dataframe):
			gene_matrix = adata.X.values
		else:
			raise ValueError(f"""{type(adata.X)} is not a valid type.""")
	else:
		gene_matrix = adata.obsm[use_data]
	weights_matrix = adata.obsm[weights]
	weights_list = []
	final_coordinates = []
	for i in range(adata.shape[0]):
		if weights == "physical_distance":
			current_spot = adata.obsm[weights][i].argsort()[-(neighbour_k+3):][:(neighbour_k+2)]
		else:
			current_spot = adata.obsm[weights][i].argsort()[-neighbour_k:][:neighbour_k-1]
		spot_weight = adata.obsm[weights][i][current_spot]
		spot_matrix = gene_matrix[current_spot]
		if spot_weight.sum() > 0:
			spot_weight_scaled = spot_weight / spot_weight.sum()
			weights_list.append(spot_weight_scaled)
			spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1, 1), spot_matrix)
			spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
		else:
			spot_matrix_final = np.zeros(gene_matrix.shape[1])
			weights_list.append(np.zeros(len(current_spot)))
		final_coordinates.append(spot_matrix_final)
	adata.obsm['adjacent_data'] = np.array(final_coordinates)
	if verbose:
		adata.obsm['adjacent_weight'] = np.array(weights_list)
	return adata
def augment_gene_data(adata, Adj_WT=0.2):
    adjacent_gene_matrix = adata.obsm["adjacent_data"].astype(float)
    if isinstance(adata.X, np.ndarray):
        augment_gene_matrix = adata.X + Adj_WT * adjacent_gene_matrix
    elif isinstance(adata.X, csr_matrix):
        augment_gene_matrix = adata.X.toarray() + Adj_WT * adjacent_gene_matrix
    adata.obsm["augment_gene_data"] = augment_gene_matrix
    del adjacent_gene_matrix
    return adata
def augment_adata(
	adata,
	platform = "Visium",
	pd_dist_type="euclidean",
	md_dist_type="cosine",
	gb_dist_type="correlation",
	n_components = 50,
	no_morphological = False,
	use_data = "raw",
	neighbour_k = 4,
	weights = "weights_matrix_all",
	Adj_WT = 0.2,
	spatial_k = 30,
	spatial_type = "KDTree"
	):
	adata = cal_weight_matrix(
				adata,
				platform = platform,
				pd_dist_type = pd_dist_type,
				md_dist_type = md_dist_type,
				gb_dist_type = gb_dist_type,
				n_components = n_components,
				no_morphological = no_morphological,
				spatial_k = spatial_k,
				spatial_type = spatial_type,
				)
	adata = find_adjacent_spot(adata,use_data = use_data,neighbour_k = neighbour_k,weights = weights)
	adata = augment_gene_data(adata,Adj_WT = Adj_WT,)
	return adata
    