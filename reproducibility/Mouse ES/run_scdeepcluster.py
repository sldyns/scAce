import h5py
import numpy as np
import scanpy as sc
import torch
from scDeepCluster import scDeepCluster
import time

from reproducibility.utils import data_sample, data_preprocess, set_seed, calculate_metric, read_data

####################################  Read dataset  ####################################

mat, obs, var, uns = read_data('../data/Mouse_E.h5', sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

####################################  Set parameters  ####################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_clusters = len(np.unique(y))


####################################  Run without sampling  ####################################

seed = 0
set_seed(seed)

adata = sc.AnnData(x)
adata.obs['celltype'] = y

adata = data_preprocess(adata, use_count=True)

model = scDeepCluster(input_dim=adata.n_vars, z_dim=32,
                      encodeLayer=[256, 64], decodeLayer=[64, 256],
                      device=device)
start = time.time()

model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors)

y_pred, _, _, _, embedded = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                      n_clusters=n_clusters, init_centroid=None,
                                      y_pred_init=None, y=y)

end = time.time()
run_time = end - start
print(f'Total time: {end - start} seconds')

nmi, ari = calculate_metric(y, y_pred)
print('Evaluating cells: NMI= %.4f, ARI= %.4f' % (nmi, ari))

np.savez("results/scDeepcluster_wo_sample.npz", ARI=ari, NMI=nmi, Embedding=embedded,
         Clusters=y_pred, Labels=y, Time_use=run_time)

####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all = [], []

for i in range(total_rounds):
    print('----------------Round: %d-------------------' % int(i + 1))
    seed = 10 * i
    set_seed(0)

    x_sample, y_sample = data_sample(x, y, seed)

    adata = sc.AnnData(x_sample)
    adata.obs['celltype'] = y_sample

    adata = data_preprocess(adata, use_count=True)

    model = scDeepCluster(input_dim=adata.n_vars, z_dim=32,
                          encodeLayer=[256, 64], decodeLayer=[64, 256],
                          device=device)

    model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors)

    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                   n_clusters=n_clusters, init_centroid=None,
                                   y_pred_init=None, y=y)

    nmi_all.append(nmi)
    ari_all.append(ari)

print(nmi_all)
print(ari_all)

np.savez("results/scDeepCluster_with_sample.npz", ARI=ari_all, NMI=nmi_all)
