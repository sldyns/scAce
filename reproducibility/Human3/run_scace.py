import h5py
import numpy as np
import scanpy as sc

from reproducibility.utils import data_sample, data_preprocess, set_seed
from scace import run_scace

####################################  Read dataset  ####################################

data_mat = h5py.File('../data/Human3.h5')
x, y = np.array(data_mat['X']), np.array(data_mat['Y'])
data_mat.close()

####################################  Run without sampling  ####################################

seed = 2023
set_seed(seed)

adata = sc.AnnData(x)
adata.obs['celltype'] = y

adata = data_preprocess(adata)
adata, nmi, ari, K, pred_all, emb_all = run_scace(adata, cl_type='celltype', n_epochs_pre=300, return_all=True)

np.savez("./results/scAce_wo_sample.npz", ARI=ari, NMI=nmi, K=K, Embedding=emb_all, Clusters=pred_all, Labels=y)

####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all, k_all, pred_all, true_all = [], [], [], [], []

for i in range(total_rounds):
    print('----------------Round: %d-------------------' % int(i + 1))
    seed = 10 * i
    set_seed(2023)

    x_sample, y_sample = data_sample(x, y, seed)

    adata = sc.AnnData(x_sample)
    adata.obs['celltype'] = y_sample

    adata = data_preprocess(adata)
    adata, nmi, ari, K, _, _ = run_scace(adata, cl_type='celltype', n_epochs_pre=300, return_all=True)

    nmi_all.append(nmi)
    ari_all.append(ari)
    k_all.append(K)
    pred_all.append(adata.obs['scace_cluster'].values.astype('int'))
    true_all.append(y_sample)

print(nmi_all)
print(ari_all)
print(k_all)

np.savez("results/scAce_with_sample.npz", ARI=ari_all, NMI=nmi_all, K=k_all, Clusters=pred_all, Labels=true_all)
