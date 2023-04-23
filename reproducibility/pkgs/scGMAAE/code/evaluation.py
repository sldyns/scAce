import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred, name=None, path=None):
    # y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, 1


def eva(y_true, y_pred, epoch=0, pp=True, name=None, path=None):
    acc, f1 = cluster_acc(y_true, y_pred, name, path)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    #nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
    ari = ari_score(y_true, y_pred)
    if pp:
        print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
    return acc, nmi, ari
