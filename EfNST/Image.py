# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:03:53 2024

@author: lenovo
"""
import random
import numpy as np 
import pandas as pd 
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn
from torch.autograd import Variable 
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from pathlib import Path
from tqdm import tqdm
class Image_Feature:
    def __init__(
        self,
        adata,
        pca_components=50,
        cnnType='efficientnet-b0',
        verbose=False,
        seeds=88,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType
    def efficientNet_model(self):
        efficientnet_versions = {
            'efficientnet-b0': 'efficientnet-b0',
            'efficientnet-b1': 'efficientnet-b1',
            'efficientnet-b2': 'efficientnet-b2',
            'efficientnet-b3': 'efficientnet-b3',
            'efficientnet-b4': 'efficientnet-b4',
            'efficientnet-b5': 'efficientnet-b5',
            'efficientnet-b6': 'efficientnet-b6',
            'efficientnet-b7': 'efficientnet-b7',
        }
        if self.cnnType in efficientnet_versions:
            model_version = efficientnet_versions[self.cnnType]
            cnn_pretrained_model = EfficientNet.from_pretrained(model_version)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(f"{self.cnnType} is not a valid EfficientNet type.")
        return cnn_pretrained_model
    def Extract_Image_Feature(self,):
    
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std =[0.229, 0.224, 0.225]),
                          transforms.RandomAutocontrast(),
                          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                          transforms.RandomInvert(),
                          transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                          transforms.RandomSolarize(random.uniform(0, 1)),
                          transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
                          transforms.RandomErasing()
                          ]
        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize(mean=[0.54, 0.51, 0.68], 
        #                   std =[0.25, 0.21, 0.16])]
        img_to_tensor = transforms.Compose(transform_list)

        feat_df = pd.DataFrame()
        model = self.efficientNet_model()
        #model.fc = torch.nn.LeakyReLU(0.1)
        model.eval()

        if "slices_path" not in self.adata.obs.keys():
             raise ValueError("Please run the function image_crop first")        
        for spot, slice_path in self.adata.obs['slices_path'].items():
            spot_slice = Image.open(slice_path)
            spot_slice = spot_slice.resize((224,224))
            spot_slice = np.asarray(spot_slice, dtype="int32")
            spot_slice = spot_slice.astype(np.float32)
            tensor = img_to_tensor(spot_slice)
            tensor = tensor.resize_(1,3,224,224)
            tensor = tensor.to(self.device)
            result = model(Variable(tensor))
            result_npy = result.data.cpu().numpy().ravel()
            feat_df[spot] = result_npy
            feat_df = feat_df.copy()   
        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat'] !")
        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
        if self.verbose:
            print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
        return self.adata 

def image_crop(
        adata,
        save_path,
        library_id=None,
        crop_size=50,
        target_size=224,
        verbose=False,
        quality='hires'):
    if library_id is None:
       library_id = list(adata.uns["spatial"].keys())[0]
       adata.uns["spatial"][library_id]["use_quality"] = quality

    image = adata.uns["spatial"][library_id]["images"][
            adata.uns["spatial"][library_id]["use_quality"]]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)
    tile_names = []

    with tqdm(total=len(adata),
              desc="Tiling image",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.ANTIALIAS) ##### 
            tile.resize((target_size, target_size)) ###### 
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)

    adata.obs["slices_path"] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path'] !")
    return adata
