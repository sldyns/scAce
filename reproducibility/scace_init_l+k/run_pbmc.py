import h5py
import numpy as np
import scanpy as sc

from reproducibility.utils import data_preprocess, set_seed
from scace import run_scace

from collections import Counter

####################################  Read dataset & Pre-process  ####################################

data_mat = h5py.File('../data/Human_PBMC.h5')
x, y = np.array(data_mat['X']), np.array(data_mat['Y'])
data_mat.close()

print(f'The size of dataset: {x.shape}')
print(f'The number of cell types: {len(np.unique(y))}')


seed = 2023
set_seed(seed)

adata = sc.AnnData(x)
adata.obs['celltype'] = y

adata = data_preprocess(adata)



######################### kmeans ############################
print('------------------------ Kmeans ----------------------------')
adata, nmi, ari, _, pred_all, emb_all, _ = run_scace(adata, cl_type='celltype', return_all=True,
                                                     init_method='kmeans')

np.savez("results/scAce_kmeans_pbmc.npz", ARI=ari, NMI=nmi, Embedding=emb_all, Clusters=pred_all, Labels=y)



######################### louvain ############################
print('------------------------ Louvain ----------------------------')
adata, nmi, ari, _, pred_all, emb_all, _ = run_scace(adata, cl_type='celltype', return_all=True,
                                                     init_method='louvain')

np.savez("results/scAce_louvain_pbmc.npz", ARI=ari, NMI=nmi, Embedding=emb_all, Clusters=pred_all, Labels=y)
