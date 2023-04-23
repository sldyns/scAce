## Version of some comparison methods

- CIDR: 0.1.5
- DESC: 2.1.1
- SCCAF: 0.0.10
- scVI: 0.17.3
- Seurat: 3.2.3

## graph-sc

graph-sc is cloned from https://github.com/ciortanmadalina/graph-sc.git

## scDeepCluster

scDeepCluster is cloned from https://github.com/ttgump/scDeepCluster_pytorch.git

## scGMAAE

scGMAAE is cloned from https://github.com/WHY-17/scGMAAE.git

## ADClust

To complete the experiment, we made some changes to the output in the source code of ADClust.

ADClust is cloned from https://github.com/biomed-AI/ADClust.git,
changed ` _adclust_training`,   ` _adclust`,  ` ADClust`.

For `_adclust_training`, we created 2 empty lists at the beginning to store the cluster labels after each interation of
merging (` pred_each_merge`) and the cluster labels after each variation of merging (` pred_all`):

```
pred_each_merge = []
pred_all = []
```

At each time the maximum value in the dip-score matrix is greater than the threshold, we stored the cluster labels at
that point in *pred_all*, and also when *m=0*, we stored the cluster labels at that point in ` pred_each_merge`:

```
while dip_matrix_cpu[dip_argmax] >= dip_merge_threshold and n_clusters_current > n_clusters_min:
	pred_all.append(np.array(cluster_labels_cpu, dtype='int'))
	if m == 0:
		pred_each_merge.append(np.array(cluster_labels_cpu, dtype='int'))
```

When `i == dedc_epochs`, we stored the cluster labels at that point in ` pred_each_merge` and ` pred_all`:

```
if i == dedc_epochs:
	pred_each_merge.append(np.array(cluster_labels_cpu, dtype='int'))
	pred_all.append(np.array(cluster_labels_cpu, dtype='int'))
```

We added `pred_each_merge` and `pred_all` to return:

```
return cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder, pred_all, pred_each_merge
```

For `_adclust`, we created 2 empty lists at the beginning to store the initial and final cluster labels and the
resulting embeddings:

```
embedded = []
clusters = []

## After pretraining add embeddings to embedded
embedded_data = encode_batchwise(testloader, autoencoder, device)
embedded.append(embedded_data)

## After obtaining the initial clustering results add them to clusters
init_centers, cluster_labels_cpu = get_center_labels(embedded_data, resolution=3.0)
clusters.append(cluster_labels_cpu)

## After obtaining the final clustering results add them to clusters and embedded
embedded_last = encode_batchwise(testloader, autoencoder, device)
embedded.append(embedded_last)
clusters.append(cluster_labels_cpu)
```

For `ADClust`, we added `pred_each_merge`, `pred_all` and `clusters` to returns:

```
return labels, n_clusters, pred_all, pred_each_merge, embedded, tsne, clusters
```
