import numpy as np
import scanpy as sc
from SCCAF import SCCAF_assessment, plot_roc, SCCAF_optimize_all

from reproducibility.utils import data_sample, data_preprocess, read_data, calculate_metric

####################################  Read dataset  ####################################

mat, obs, var, uns = read_data('../data/Mouse2.h5', sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

####################################  Run without sampling  ####################################

pred_all = []
adata = sc.AnnData(x)
adata.obs['celltype'] = y
adata = data_preprocess(adata)

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.leiden(adata, key_added='louvain_r1')
sc.tl.tsne(adata, random_state=0)
y_pred_init = np.array(adata.obs['louvain_r1'], dtype=int).squeeze()
pred_all.append(y_pred_init)

adata.obs['SCCAF_pred_init'] = adata.obs['louvain_r1']
adata.obs['SCCAF_true_init'] = adata.obs['celltype']

y_prob, y_pred, y_test, clf, cvsm, acc = SCCAF_assessment(adata.X, adata.obs['celltype'])
aucs = plot_roc(y_prob, y_test, clf, cvsm=cvsm, acc=acc)
adata.obs['L1_Round0'] = adata.obs['louvain_r1']
SCCAF_optimize_all(ad=adata, basis='tsne', use='pca')  # use='pca'

pred_merge = adata.obs.loc[:, lambda d: d.columns.str.contains('Round')]
pred_merge = pred_merge.iloc[:, [i % 2 == 0 for i in range(len(pred_merge.columns))]]
pred_all.append(pred_merge)

y_pred_last = np.array(adata.obs['L1_result'], dtype=int).squeeze()
pred_all.append(y_pred_last)
nmi, ari = calculate_metric(adata.obs['celltype'], adata.obs['L1_result'])

K = len(np.unique(np.asarray(adata.obs['L1_result'], dtype='int')))

np.savez("./results/SCCAF_wo_sample.npz", ARI=ari, NMI=nmi, K=K, Embedding=np.array(adata.X), Clusters=pred_all,
         pca=np.array(adata.obsm['X_pca']))

####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all, k_all, pred_all, true_all = [], [], [], [], []

for i in range(total_rounds):
    print('----------------Rounds: %d-------------------' % int(i + 1))

    seed = 10 * i
    x_sample, y_sample = data_sample(x, y, seed)
    adata = sc.AnnData(x_sample)
    adata.obs['celltype'] = y_sample

    adata = data_preprocess(adata)

    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, key_added='louvain_r1')
    sc.tl.tsne(adata)
    adata.obs['SCCAF_pred_init'] = adata.obs['louvain_r1']
    adata.obs['SCCAF_true_init'] = adata.obs['celltype']

    y_prob, y_pred, y_test, clf, cvsm, acc = SCCAF_assessment(adata.X, adata.obs['celltype'])
    aucs = plot_roc(y_prob, y_test, clf, cvsm=cvsm, acc=acc)
    adata.obs['L1_Round0'] = adata.obs['louvain_r1']
    SCCAF_optimize_all(ad=adata, basis='tsne', use='pca')  # use='pca'
    nmi, ari = calculate_metric(adata.obs['celltype'], adata.obs['L1_result'])
    K = len(np.unique(np.asarray(adata.obs['L1_result'], dtype='int')))

    print(ari)
    print(nmi)
    print(K)

    #     ACC_F.append(ACC)
    nmi_all.append(nmi)
    ari_all.append(ari)
    k_all.append(K)
    pred_all.append(np.asarray(adata.obs['L1_result'], dtype='int'))
    true_all.append(np.asarray(adata.obs['celltype'], dtype='int'))

# print(f'ACC: {ACC_F}')
print(f'ARI: {ari_all}')
print(f'NMI: {nmi_all}')
print(k_all)

np.savez("./results/SCCAF_with_sample.npz", ARI=ari_all, NMI=nmi_all, K=k_all, Clusters=pred_all, Labels=true_all)
