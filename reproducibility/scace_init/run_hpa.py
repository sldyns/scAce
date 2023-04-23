import h5py
import numpy as np
import scanpy as sc

from reproducibility.utils import data_preprocess, set_seed
from scace import run_scace

####################################  Read dataset & Pre_process  ####################################

data_mat = h5py.File('../data/Human_p.h5')
x, y = np.array(data_mat['X']), np.array(data_mat['Y'])
data_mat.close()

print(f'The size of dataset: {x.shape}')
print(f'The number of cell types: {len(np.unique(y))}')


seed = 2023
set_seed(seed)

adata = sc.AnnData(x)
adata.obs['celltype'] = y

adata = data_preprocess(adata)
adata, nmi, ari, K, pred_all, emb_all = run_scace(adata, cl_type='celltype', return_all=True,
                                                  resolution=0.85)
# adata = run_scace(adata)

emb_init = np.array(emb_all[0])
pred_init = np.array(pred_all[0])

np.savez("results/scAce_init_human.npz", Embedding=emb_init, Clusters=pred_init, Labels=y)