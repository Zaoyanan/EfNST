# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:28:11 2024

@author: lenovo
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type("Literal_", (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass
_QUALITY = Literal["fulres", "hires", "lowres"]
_background = ["black", "white"]
#class VisiumDataProcessor:
def read_Visium(path, genome=None, count_file='filtered_feature_bc_matrix.h5', 
                    library_id=None, load_images=True, quality='hires', image_path=None):
    adata = sc.read_visium(path, 
                    genome=genome,
                    count_file=count_file,
                    library_id=library_id,
                    load_images=load_images,)
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = (adata.obsm["spatial"])
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata

class Refiner:
    def __init__(self, shape="hexagon"):
        self.shape = shape
        self.pred_df = None
        self.dis_df = None
    def fit(self, sample_id, pred, dis):
        self.pred_df = pd.DataFrame({"pred": pred}, index=sample_id)
        self.dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    def get_neighbors(self, index, num_neighbors):
        distances = self.dis_df.loc[index, :].sort_values()
        return distances.index[1:num_neighbors+1]
    def majority_vote(self, predictions):
        counts = np.bincount(predictions)
        return np.argmax(counts)
    def refine(sample_id, pred, dis, shape="hexagon"):
        refined_pred = []
        pred=pd.DataFrame({"pred": pred}, index=sample_id)
        dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
        if shape == "hexagon":
            num_nbs = 6 
        elif shape == "square":
            num_nbs = 4
        for i in range(len(sample_id)):
            index=sample_id[i]
            dis_tmp=dis_df.loc[index, :].sort_values()
            nbs=dis_tmp[0:num_nbs+1]
            nbs_pred=pred.loc[nbs.index, "pred"]
            self_pred=pred.loc[index, "pred"]
            v_c=nbs_pred.value_counts()
            if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
                refined_pred.append(v_c.idxmax())
            else:           
                refined_pred.append(self_pred)
        return refined_pred
