# EfNST: A composite scaling network of EfficientNet for improving spatial domain identification performance
Spatial Transcriptomics (ST) leverages Gene Expression Profiling while preserving Spatial Location and Histological Images, enabling it to provide new insights into tissue structure, tumor microenvironment, and biological development. However, challenges persist due to significant image noise when utilizing images for spatial recognition. Here, we propose EfNST, which firstly introduce an efficient composite scaling network EfficientNet for processing image data, enabling the model to effectively learn multi-scale image features. We applied EfNST to four different tissue types of the 10X Visium sequencing platform, and it outperforms five advanced competing algorithms, achieving the best Adjusted Rand Index (ARI) scores in spatial recognition. EfNST showcased the ability to finely identify subregions of tumors in a human breast cancer data set. In an unannotated mouse brain data set, EfNST not only identified tiny regions of complex tissues but also resolved their spatial expression patterns in biological processes. Additionally, EfNST demonstrated high accuracy in identifying fine tissue structures and discovering corresponding marker genes with an improved running speed. In conclusion, EfNST offers a novel approach to inferring cellular spatial organization from discrete data spots, with its significance extending to the exploration of tissue structure, function, and organism development
## Overview
![image](https://github.com/Zaoyanan/EfNST/blob/main/figure/Overview.png)
Fig. 1. Workflow of the EfNST. (a) The input ST data are Gene Expression, Histological Images and Spatial Location; (b) EfNST processed the H&E Images and Spatial Locations to obtain Image Patches, which were processed using a pre-trained EfficientNet network to obtain Image Feature Matrix. Data Augmentation is performed for each spot based on the similarity of spots in spatial combined with the gene expression weights and the spatial location weights; (c) Final latent embedding is achieved using VGAE and DAE, H1 represtents Hidden layer, H2 represtents Low dimensional representation; (d) Latent representations can be used to perform Downstream Analysis.
## Dependencies
- Python=3.9.16
- numpy=1.22.4
- pandas=1.5.2
- matplotlib=3.6.3
- scanpy=1.9.1
- torch=2.0.1
- torch_geometric=2.2.1
- Pillow=0.5.0
- efficientnet-pytorch=0.7.1
- igraph=0.10.4
- leidenalg=0.9.1
- scikit-learn=1.2.2
- networkx=2.7
- scipy=1.8.0
## Tutorial
Check the Tutorial folder for detailed instructions.
