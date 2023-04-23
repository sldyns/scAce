import h5py
import numpy as np
import scanpy as sc
from ADClust import ADClust
import time

from reproducibility.utils import data_sample, data_preprocess, set_seed, calculate_metric, read_data

####################################  Read dataset  ####################################

mat, obs, var, uns = read_data('../data/Mouse_k.h5', sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

####################################  Run without sampling  ####################################

adata = sc.AnnData(x)
adata.obs['CellType'] = y
adata.obs['CellType'] = adata.obs['CellType'].astype(str).astype('category')

adata = data_preprocess(adata, select_gene_adclust=True)
data = adata.X
labels = adata.obs['CellType'].values.astype(np.int32)

set_seed(0)
adClust = ADClust(data_size=data.shape[0])

start = time.time()
cluster_labels, estimated_cluster_numbers, pred_all, pred, embedded, clusters = adClust.fit(data)
print("The estimated number of clusters:", estimated_cluster_numbers)

end = time.time()
run_time = end - start
print(f'Total time: {end - start} seconds')

nmi, ari = calculate_metric(labels, cluster_labels)
K = estimated_cluster_numbers
print("ARI: ", ari)
print("NMI:", nmi)

np.savez("results/ADClust_wo_sample.npz", ARI=ari, NMI=nmi, K=K, Embedding=embedded,
         Clusters=cluster_labels, Clusters_merge=pred, Clusters_merge_all=pred_all, Time_use=run_time)

####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all, k_all, pred_all, true_all = [], [], [], [], []

for i in range(total_rounds):
    print('----------------Rounds: %d-------------------' % int(i + 1))

    set_seed(0)

    seed = 10 * i
    x_sample, y_sample = data_sample(x, y, seed)

    adata = sc.AnnData(x_sample)
    adata.obs['CellType'] = y_sample
    adata.obs['CellType'] = adata.obs['CellType'].astype(str).astype('category')
    adata = data_preprocess(adata, select_gene_adclust=True)
    data = adata.X
    labels = adata.obs['CellType'].values.astype(np.int32)

    adClust = ADClust(data_size=data.shape[0])
    cluster_labels, estimated_cluster_numbers, _, _, _, _ = adClust.fit(data)
    print("The estimated number of clusters:", estimated_cluster_numbers)

    nmi, ari = calculate_metric(labels, cluster_labels)
    K = estimated_cluster_numbers
    print("ARI: ", ari)
    print("NMI:", nmi)

    nmi_all.append(nmi)
    ari_all.append(ari)
    k_all.append(K)
    pred_all.append(np.asarray(cluster_labels, dtype='int'))
    true_all.append(y_sample)

print(f'ARI: {ari_all}')
print(f'NMI: {nmi_all}')
print(K)

np.savez("results/ADClust_with_sample.npz", ARI=ari_all, NMI=nmi_all, K=k_all, Clusters=pred_all, Labels=true_all)
