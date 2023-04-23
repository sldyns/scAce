import numpy as np
import scanpy as sc

from reproducibility.utils import data_preprocess, set_seed, read_data
from scace import run_scace

####################################  Read dataset & Pre-process  ####################################

mat, obs, var, uns = read_data('../data/Mouse_h.h5', sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

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

np.savez("results/scAce_kmeans_chen.npz", ARI=ari, NMI=nmi, Embedding=emb_all, Clusters=pred_all, Labels=y)



######################### louvain ############################
print('------------------------ Louvain ----------------------------')
adata, nmi, ari, _, pred_all, emb_all, _ = run_scace(adata, cl_type='celltype', return_all=True,
                                                     init_method='louvain')

np.savez("results/scAce_louvain_chen.npz", ARI=ari, NMI=nmi, Embedding=emb_all, Clusters=pred_all, Labels=y)
