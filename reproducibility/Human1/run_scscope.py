import h5py
import numpy as np
import phenograph
import scanpy as sc
import scscope as DeepImpute

from reproducibility.utils import data_sample, tf_seed, calculate_metric

####################################  Read dataset  ####################################

data_mat = h5py.File('../data/Human1.h5')
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])
data_mat.close()

gene_expression = sc.AnnData(x)
sc.pp.normalize_per_cell(gene_expression)
gene_expression = gene_expression.X

####################################  Run without sampling  ####################################

tf_seed(0)

latent_dim = 50
DI_model = DeepImpute.train(gene_expression, latent_dim, epoch_per_check=10)
latent_code, imputed_val, _ = DeepImpute.predict(gene_expression, DI_model)

y_pred, _, _ = phenograph.cluster(latent_code, clustering_algo='leiden', seed=0)

nmi, ari = calculate_metric(y, y_pred)

k = len(np.unique(y_pred))
embedded = np.array(latent_code)

print('Clustering PhenoGraph: ARI= %.4f, NMI= %.4f' % (ari, nmi))
np.savez("./results/scScope_wo_sample.npz", ARI=ari, NMI=nmi, K=k, Embedding=embedded, Clusters=y_pred)

####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all, k_all, pred_all, true_all = [], [], [], [], []

for i in range(total_rounds):
    print('----------------Round: %d-------------------' % int(i + 1))

    seed = 10 * i
    x_sample, y_sample = data_sample(x, y, seed)

    gene_expression = sc.AnnData(x_sample)
    sc.pp.normalize_per_cell(gene_expression)
    gene_expression = gene_expression.X

    latent_dim = 50
    DI_model = DeepImpute.train(gene_expression, latent_dim, epoch_per_check=10)
    latent_code, imputed_val, _ = DeepImpute.predict(gene_expression, DI_model)

    y_pred, _, _ = phenograph.cluster(latent_code, clustering_algo='leiden', seed=0)

    nmi, ari = calculate_metric(y_sample, y_pred)
    k = len(np.unique(y_pred))
    print('Clustering PhenoGraph: ARI= %.4f, NMI= %.4f' % (ari, nmi))

    nmi_all.append(nmi)
    ari_all.append(ari)
    k_all.append(k)
    pred_all.append(np.asarray(y_pred, dtype='int'))
    true_all.append(y_sample)

print(nmi_all)
print(ari_all)
print(k_all)

np.savez("results/scScope_with_sample.npz", ARI=ari_all, NMI=nmi_all, K=k_all, Clusters=pred_all, Labels=true_all)
