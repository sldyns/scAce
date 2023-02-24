import numpy as np
import pandas as pd
import scanpy as sc

from reproducibility.utils import data_preprocess, set_seed, read_data
from scace import run_scace

####################################  Read data and previous clustering results  ####################################

mat, obs, var, uns = read_data('../data/Mouse2.h5', sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

y_pred_init = pd.read_csv('./pred/CIDR/Mouse2_CIDR_pred.csv', header=None).values.reshape(-1).astype('int')
# Translate label to 0 ~ len(np.unique(label))-1
for i in range(len(np.unique(y_pred_init))):
    y_pred_init[y_pred_init == np.unique(y_pred_init)[i]] = i

####################################  Enhance CIDR's result  #######################################

seed = 2023
set_seed(seed)

adata = sc.AnnData(x)
adata.obs['celltype'] = y

adata = data_preprocess(adata)
adata, nmi, ari, K, pred_all, emb_all = \
    run_scace(adata, cl_type='celltype', n_epochs_pre=300, init_cluster=y_pred_init, return_all=True)

embedding_all, clusters_all = [], []
embedding_all.append(emb_all[0])
embedding_all.append(adata.obsm['scace_emb'])
clusters_all.append(pred_all[0])
clusters_all.append(adata.obs['scace_cluster'].values.astype('int'))

np.savez("./results/scAce_enhance_CIDR_mouse2.npz", ARI=ari, NMI=nmi, K=K, Embedding=embedding_all,
         Clusters=clusters_all, Labels=y)
