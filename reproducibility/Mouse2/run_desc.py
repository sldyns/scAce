import desc as desc
import numpy as np
import scanpy as sc

from reproducibility.utils import data_sample, data_preprocess, set_seed, calculate_metric, read_data

sc.settings.verbosity = 3
sc.logging.print_versions()

####################################  Read dataset  ####################################

mat, obs, var, uns = read_data('../data/Mouse2.h5', sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

####################################  Run without sampling  ####################################

adata = sc.AnnData(x)
adata.obs['celltype'] = y
adata.obs['celltype'] = adata.obs['celltype'].astype(str).astype('category')

adata = data_preprocess(adata, scale_factor=False, counts_per_cell=True,
                        normalize_input=False, select_gene_desc=True)
desc.scale(adata, zero_center=True, max_value=3)

set_seed(0)
adata = desc.train(adata,
                   dims=[adata.shape[1], 64, 32],
                   tol=0.005,
                   pretrain_epochs=300,
                   louvain_resolution=[0.8],
                   save_dir="result_pbmc3k",
                   use_ae_weights=False,
                   do_tsne=True,
                   do_umap=False,
                   use_GPU=True,
                   num_Cores=1,
                   num_Cores_tsne=4,
                   save_encoder_weights=False)

y_pred = np.asarray(adata.obs['desc_0.8'], dtype=int)
embedding = np.array(adata.obsm['X_Embeded_z0.8'])
# tsne = np.array(adata.obsm['X_tsne0.8'])
k = len(np.unique(y_pred))
nmi, ari = calculate_metric(y, y_pred)

print(ari)
print(nmi)
print(k)

np.savez("results/DESC_wo_sample.npz", ARI=ari, NMI=nmi, K=k, Embedding=embedding, Clusters=y_pred)

####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all, k_all, pred_all, true_all = [], [], [], [], []

for i in range(total_rounds):
    print('----------------Rounds: %d-------------------' % int(i + 1))

    seed = 10 * i
    x_sample, y_sample = data_sample(x, y, seed)

    adata = sc.AnnData(x_sample)
    adata.obs['celltype'] = y_sample
    adata.obs['celltype'] = adata.obs['celltype'].astype(str).astype('category')

    adata = data_preprocess(adata, scale_factor=False, counts_per_cell=True,
                            normalize_input=False, select_gene_desc=True)
    desc.scale(adata, zero_center=True, max_value=3)

    set_seed(0)
    adata = desc.train(adata,
                       dims=[adata.shape[1], 64, 32],
                       tol=0.005,
                       pretrain_epochs=300,
                       louvain_resolution=[0.8],
                       save_dir="result_pbmc3k",
                       use_ae_weights=False,
                       do_tsne=True,
                       do_umap=False,
                       use_GPU=True,
                       num_Cores=1,
                       num_Cores_tsne=4,
                       save_encoder_weights=False)

    y_pred = np.asarray(adata.obs['desc_0.8'], dtype=int)
    embedding = np.array(adata.obsm['X_Embeded_z0.8'])
    k = len(np.unique(y_pred))
    nmi, ari = calculate_metric(y_sample, y_pred)

    ari_all.append(ari)
    nmi_all.append(nmi)
    k_all.append(k)
    pred_all.append(np.asarray(y_pred, dtype='int'))
    true_all.append(y_sample)

    print(ari)
    print(nmi)
    print(k)

print(f'ARI: {ari_all}')
print(f'NMI: {nmi_all}')
print(k_all)

np.savez("results/DESC_with_sample.npz", ARI=ari_all, NMI=nmi_all, K=k_all, Clusters=pred_all, Labels=true_all)
