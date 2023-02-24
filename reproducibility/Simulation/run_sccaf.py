import h5py
import numpy as np
import scanpy as sc
from SCCAF import SCCAF_assessment, plot_roc, SCCAF_optimize_all

from reproducibility.utils import data_sample, data_preprocess, calculate_metric

####################################  Read dataset  ####################################

data_mat = h5py.File('../data/Sim.h5')
x, y = np.array(data_mat['X']), np.array(data_mat['Y'])
data_mat.close()

####################################  Run without sampling  ####################################

adata = sc.AnnData(x)
adata.obs['celltype'] = y
adata = data_preprocess(adata)

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.leiden(adata, key_added='louvain_r1')
sc.tl.tsne(adata, random_state=0)
y_pred_init = np.array(adata.obs['louvain_r1'], dtype=int).squeeze()

adata.obs['SCCAF_pred_init'] = adata.obs['louvain_r1']
adata.obs['SCCAF_true_init'] = adata.obs['celltype']

y_prob, y_pred, y_test, clf, cvsm, acc = SCCAF_assessment(adata.X, adata.obs['celltype'])
aucs = plot_roc(y_prob, y_test, clf, cvsm=cvsm, acc=acc)
adata.obs['L1_Round0'] = adata.obs['louvain_r1']
SCCAF_optimize_all(ad=adata, basis='tsne', use='pca')  # use='pca'

y_pred_last = np.array(adata.obs['L1_result'], dtype=int).squeeze()
nmi, ari = calculate_metric(adata.obs['celltype'], adata.obs['L1_result'])

K = len(np.unique(np.asarray(adata.obs['L1_result'], dtype='int')))

np.savez("./results/SCCAF_wo_sample.npz", ARI=ari, NMI=nmi, K=K, Embedding=np.array(adata.X), Clusters=y_pred_last)

####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all, k_all = [], [], []

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

    nmi_all.append(nmi)
    ari_all.append(ari)
    k_all.append(K)

print(f'ARI: {ari_all}')
print(f'NMI: {nmi_all}')
print(k_all)

np.savez("./results/SCCAF_with_sample.npz", ARI=ari_all, NMI=nmi_all, K=k_all)
