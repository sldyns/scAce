import h5py
import numpy as np
import pandas as pd
import scanpy as sc

from scace import run_scace
from reproducibility.utils import data_preprocess, set_seed

####################################  Read data and previous clustering results  ####################################

data_mat = h5py.File('dataset/Human_p.h5')
x, y = np.array(data_mat['X']), np.array(data_mat['Y'])
data_mat.close()

y_pred_init = pd.read_csv('pred/Seurat/Human.csv', header=None).values.reshape(-1).astype('int')
# Translate label to 0 ~ len(np.unique(label))-1
for i in range(len(np.unique(y_pred_init))):
    y_pred_init[y_pred_init == np.unique(y_pred_init)[i]] = i

####################################  Enhance Seurat's result  #######################################

seed = 0
set_seed(seed)

adata = sc.AnnData(x)
adata.obs['celltype'] = y

adata = data_preprocess(adata)

cell_index = np.array(adata.obs_names)
for i in range(len(cell_index)):
    cell_index[i] = int(cell_index[i])
cell_index = list(cell_index)
y_pred_init = y_pred_init[[cell_index]].squeeze()

adata, nmi, ari, K, pred_all, emb_all, _ = \
    run_scace(adata, cl_type='celltype', init_cluster=y_pred_init, return_all=True)

embedding_all, clusters_all = [], []
embedding_all.append(emb_all[0])
embedding_all.append(adata.obsm['scace_emb'])
clusters_all.append(pred_all[0])
clusters_all.append(adata.obs['scace_cluster'].values.astype('int'))

np.savez("results/scAce_enhance_Seurat_human.npz", ARI=ari, NMI=nmi, K=K, Embedding=embedding_all,
         Clusters=clusters_all, Labels=y)