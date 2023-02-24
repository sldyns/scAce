import h5py
import numpy as np
import scanpy as sc
from ADClust import ADClust

from reproducibility.utils import data_sample, data_preprocess, set_seed, calculate_metric

####################################  Read dataset  ####################################

data_mat = h5py.File('../data/Sim.h5')
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])

####################################  Run without sampling  ####################################

adata = sc.AnnData(x)
adata.obs['CellType'] = y
adata.obs['CellType'] = adata.obs['CellType'].astype(str).astype('category')

adata = data_preprocess(adata, select_gene_adclust=True)
data = adata.X
labels = adata.obs['CellType'].values.astype(np.int32)

set_seed(0)
adClust = ADClust(data_size=data.shape[0])
cluster_labels, estimated_cluster_numbers, _, _, embedded, _ = adClust.fit(data)
print("The estimated number of clusters:", estimated_cluster_numbers)

nmi, ari = calculate_metric(labels, cluster_labels)
K = estimated_cluster_numbers
print("ARI: ", ari)
print("NMI:", nmi)

np.savez("results/ADClust_wo_sample.npz", ARI=ari, NMI=nmi, K=K, Embedding=embedded, Clusters=cluster_labels)

####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all, k_all = [], [], []

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

print(f'ARI: {ari_all}')
print(f'NMI: {nmi_all}')
print(K)

np.savez("results/ADClust_with_sample.npz", ARI=ari_all, NMI=nmi_all, K=k_all)
