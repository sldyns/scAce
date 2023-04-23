rm(list=ls())

############################### library ###############################

library(Seurat)
library(rhdf5)
library(aricode) # for calculating ARI and NMI


########################### Run without sampling ##############################

h=H5Fopen("../data/Human_PBMC.h5")
h5dump(h,load=FALSE)
raw_count=h$X
sctype= h$Y

c_names <- colnames(raw_count, do.NULL = FALSE)
r_names <- rownames(raw_count, do.NULL = FALSE)
colnames(raw_count) <- c_names
rownames(raw_count) <- r_names

start_time <- proc.time()

adata.data <- raw_count
adata <- CreateSeuratObject(counts=adata.data, min.cells=3, min.features=200)
adata <- NormalizeData(adata)
adata <- FindVariableFeatures(adata)
adata <- ScaleData(adata)
adata <- RunPCA(adata, features = VariableFeatures(object = adata))  
adata <- FindNeighbors(adata)
adata <- FindClusters(adata, random.seed =0)
adata <- RunUMAP(adata, seed.use = 0)

end_time <- proc.time()
Time_use <- end_time - start_time

cluster_l = adata@meta.data[["seurat_clusters"]]
cluster_l = as.numeric(cluster_l)
Cluster = as.matrix(cluster_l, ncol=1)

ari = ARI(c(cluster_l), c(sctype))
nmi = NMI(c(cluster_l), c(sctype))
K = length(unique(cluster_l))
UMAP = adata@reductions[["umap"]]@cell.embeddings

results = data.frame(ari, nmi, K, Cluster, UMAP, Time_use[3])
write.csv(results, "results/Seurat_wo_sample.csv")




##################### Run 10 rounds with sampling 95% data ####################

ari_all = c()
nmi_all = c()
K = c()

directory<-"../data/sample/Human_PBMC"
files <- list.files(directory, pattern = "^Human_PBMC_.*\\.h5$", full.names = TRUE)

for(i in 1:10){
  
  h=H5Fopen(files[i])
  h5dump(h,load=FALSE)
  raw_count=h$X
  sctype= h$Y
  
  c_names <- colnames(raw_count, do.NULL = FALSE)
  r_names <- rownames(raw_count, do.NULL = FALSE)
  colnames(raw_count) <- c_names
  rownames(raw_count) <- r_names

  meta.data=data.frame(celltype=sctype)
  rownames(meta.data) = c_names

  adata.data <- raw_count
  adata <- CreateSeuratObject(counts=adata.data, meta.data=meta.data)

  if(i == 1){
    clusters = matrix(NA, nrow = dim(adata@assays$RNA)[2], ncol=10, byrow = FALSE)
    labels = matrix(NA, nrow = dim(adata@assays$RNA)[2], ncol=10, byrow = FALSE)
  }
  
  adata <- NormalizeData(adata)
  adata <- FindVariableFeatures(adata, verbose = FALSE)
  adata <- ScaleData(adata)

  adata <- RunPCA(adata, features = VariableFeatures(object = adata))  
  adata <- FindNeighbors(adata)
  adata <- FindClusters(adata, random.seed =0)
  
  cluster_l = adata@meta.data[["seurat_clusters"]]
  cluster_l = as.numeric(cluster_l)
  cluster = as.matrix(cluster_l, ncol=1)

  if(length(cluster_l) < length(clusters_ll)){

    clusters_ll[1:length(cluster_l)] = cluster_l
    clusters_ll[(length(cluster_l)+1):length(clusters_ll)] = NA

    labels_ll[1:length(cluster_l)] = adata@meta.data$celltype
    labels_ll[(length(cluster_l)+1):length(clusters_ll)] = NA

    clusters[,i] = clusters_ll
    labels[,i] = labels_ll
  }

  else{
    clusters[,i] = cluster_l
    labels[,i] = adata@meta.data$celltype
  }
  
  ari_all[i] = ARI(c(cluster_l), c(sctype))
  nmi_all[i] = NMI(c(cluster_l), c(sctype))
  K[i] = length(unique(cluster_l))
}

results_list = list(ari_all, nmi_all, K, clusters, labels)
results = do.call(cbind, lapply(lapply(results_list, unlist), `length<-`, max(lengths(results_list))))
colnames(results) = c('ARI', 'NMI', 'K', rep('clusters', 10), rep('labels', 10))
write.csv(results, "results/Seurat_with_sample.csv")
