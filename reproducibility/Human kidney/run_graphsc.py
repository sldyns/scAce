import time

import graph_sc.train as train
import h5py
import numpy as np

from reproducibility.utils import data_sample, set_seed, calculate_metric

set_seed(0)

device = train.get_device(use_cpu=False)
print(f"Running on device: {device}")

####################################  Read dataset  ####################################

data_mat = h5py.File('../data/Human_k.h5')
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])

n_clusters = len(np.unique(y))

####################################  Run without sampling  ####################################

start = time.time()
scores = train.fit(x, y, n_clusters, cluster_methods=["KMeans"])

end = time.time()
run_time = end - start
print(f'Total time: {end - start} seconds')

embeddings = np.array(scores["features"])
y_pred = scores["kmeans_pred"]
nmi, ari = calculate_metric(y, y_pred)

np.savez("results/graphsc_wo_sample.npz", ARI=ari, NMI=nmi, Embedding=embeddings, Clusters=y_pred, Time_use=run_time)

#############################  Run 10 rounds with sampling 95% data  ###################################

total_rounds = 10
ari_all, nmi_all = [], []

for i in range(total_rounds):
    print('----------------Round: %d-------------------' % int(i + 1))

    seed = 10 * i
    x_sample, y_sample = data_sample(x, y, seed)

    set_seed(0)

    n_clusters = len(np.unique(y_sample))

    scores = train.fit(x_sample, y_sample, n_clusters, cluster_methods=["KMeans"])

    y_pred = scores["kmeans_pred"]
    nmi, ari = calculate_metric(y_sample, y_pred)

    ari_all.append(ari)
    nmi_all.append(nmi)

print(ari_all)
print(nmi_all)

np.savez("results/graphsc_with_sample.npz", ARI=ari_all, NMI=nmi_all)
