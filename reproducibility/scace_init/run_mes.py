import numpy as np
import scanpy as sc

from reproducibility.utils import data_preprocess, set_seed, read_data
from scace import run_scace

####################################  Read dataset & Pre_process  ####################################

mat, obs, var, uns = read_data('../data/Mouse_E.h5', sparsify=False, skip_exprs=False)
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
adata, nmi, ari, K, pred_all, emb_all = run_scace(adata, cl_type='celltype', return_all=True,
                                                  resolution=0.075)
# adata = run_scace(adata)

emb_init = np.array(emb_all[0])
pred_init = np.array(pred_all[0])

np.savez("results/scAce_init_klein.npz", Embedding=emb_init, Clusters=pred_init, Labels=y)
