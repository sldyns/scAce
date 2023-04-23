import time

import h5py
import numpy as np
import scanpy as sc
import scvi
from reproducibility.utils import data_sample, set_seed, calculate_metric, read_data


####################################  Read dataset  ####################################

mat, obs, var, uns = read_data('../data/Turtle_b.h5', sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

####################################  Run without sampling  ####################################

adata = sc.AnnData(x)
adata.obs['cell_type'] = y
adata.obs['cell_type'] = adata.obs['cell_type'].astype(str).astype('category')

adata.raw = adata
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["counts"] = adata.X.copy()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="counts",n_top_genes=2000, subset=True)

set_seed(0)
start = time.time()

scvi.model.SCVI.setup_anndata(adata,layer="counts")
vae = scvi.model.SCVI(adata)
vae.train()
adata.obsm["X_scVI"] = vae.get_latent_representation()

sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata)
pred = np.array(adata.obs['leiden'])

end = time.time()
run_time = end - start
print(f'Total time: {end - start} seconds')

embedding = np.array(adata.obsm["X_scVI"])
k = len(np.unique(pred))
nmi, ari = calculate_metric(pred, np.array(adata.obs['cell_type']).squeeze())

print("ARI: ", ari)
print("NMI:", nmi)
print("k:", k)

np.savez("results/scVI_wo_sample.npz", ARI=ari, NMI=nmi, K=k, Clusters=pred, Embedding=embedding, Time_use=run_time)


####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all, k_all, pred_all, true_all = [], [], [], [], []

for i in range(total_rounds):
    print('----------------Rounds: %d-------------------' % int(i + 1))

    set_seed(0)

    seed = 10 * i
    x_sample, y_sample = data_sample(x, y, seed)

    adata = sc.AnnData(x_sample)
    adata.obs['cell_type'] = y_sample
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str).astype('category')

    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="counts", n_top_genes=2000, subset=True)

    set_seed(0)

    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    vae = scvi.model.SCVI(adata)
    vae.train()
    adata.obsm["X_scVI"] = vae.get_latent_representation()

    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.leiden(adata)
    pred = np.array(adata.obs['leiden'])

    embedding = np.array(adata.obsm["X_scVI"])
    k = len(np.unique(pred))
    nmi, ari = calculate_metric(pred, np.array(adata.obs['cell_type']).squeeze())
    ari_all.append(ari)
    nmi_all.append(nmi)
    k_all.append(k)
    pred_all.append(pred)
    true_all.append(np.array(adata.obs['cell_type']))

    print(ari)
    print(nmi)

print(ari_all)
print(nmi_all)
print(k_all)

np.savez("results/scVI_with_sample.npz", ARI=ari_all, NMI=nmi_all, K=k_all, Clusters=pred_all, Labels=true_all)