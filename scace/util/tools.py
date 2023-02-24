import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from ..model import centroid_split


def compute_mu(scace_emb, pred):
    mu = []
    for idx in np.unique(pred):
        mu.append(scace_emb[idx == pred, :].mean(axis=0))

    return np.array(mu)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = pd.Series(data=y_pred)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    D = int(D)
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def clustering(model, exp_mat, init_cluster=None, resolution=None):
    model.eval()
    scace_emb = model.EncodeAll(exp_mat)
    model.train()

    if resolution:
        adata_l = sc.AnnData(scace_emb.cpu().numpy())
        sc.pp.neighbors(adata_l, n_neighbors=10)
        sc.tl.leiden(adata_l, resolution=resolution, random_state=0)
        y_pred = np.asarray(adata_l.obs['leiden'], dtype=int)
        mu = compute_mu(scace_emb.cpu().numpy(), y_pred)

        return y_pred, mu, scace_emb.cpu().numpy()

    if init_cluster is not None:
        cluster_centers = compute_mu(scace_emb.cpu().numpy(), init_cluster)

        data_1 = np.concatenate([scace_emb.cpu().numpy(), np.array(init_cluster).reshape(-1, 1)], axis=1)
        mu, y_pred = centroid_split(scace_emb.cpu().numpy(), data_1, cluster_centers, np.array(init_cluster))

        return y_pred, mu, scace_emb.cpu().numpy()

    # Deep Embedded Clustering
    q = model.soft_assign(scace_emb)
    p = model.target_distribution(q)

    y_pred = torch.argmax(q, dim=1).cpu().numpy()

    return y_pred, scace_emb.cpu().numpy(), q, p



def calculate_metric(pred, label):
    # acc = np.round(cluster_acc(label, pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 5)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 5)

    return nmi, ari


