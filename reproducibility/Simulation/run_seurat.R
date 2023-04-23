rm(list=ls())

############################### library ###############################

library(Seurat)
library(rhdf5)
library(aricode) # for calculating ARI and NMI


########################### Run without sampling ##############################

h=H5Fopen("../data/Sim.h5")
h5dump(h,load=FALSE)
raw_count=h$X
sctype= h$Y

c_names <- colnames(raw_count, do.NULL = FALSE)
r_names <- rownames(raw_count, do.NULL = FALSE)
colnames(raw_count) <- c_names
rownames(raw_count) <- r_names

adata.data <- raw_count
adata <- CreateSeuratObject(counts=adata.data, min.cells=3, min.features=200)
adata <- NormalizeData(adata)
adata <- FindVariableFeatures(adata)
adata <- ScaleData(adata)
adata <- RunPCA(adata, features = VariableFeatures(object = adata))  
adata <- FindNeighbors(adata)
adata <- FindClusters(adata, random.seed =0)
adata <- RunUMAP(adata, seed.use = 0)

cluster_l = adata@meta.data[["seurat_clusters"]]
cluster_l = as.numeric(cluster_l)
Cluster = as.matrix(cluster_l, ncol=1)

ari = ARI(c(cluster_l), c(sctype))
nmi = NMI(c(cluster_l), c(sctype))
K = length(unique(cluster_l))
UMAP = adata@reductions[["umap"]]@cell.embeddings

results = data.frame(ari, nmi, K, Cluster, UMAP)
write.csv(results, "results/Seurat_wo_sample.csv")




##################### Run 10 rounds with sampling 95% data ####################

ari_all = c()
nmi_all = c()
K = c()

directory<-"../data/sample/Sim"
files <- list.files(directory, pattern = "^Sim_.*\\.h5$", full.names = TRUE)

for(i in 1:10){
  
  h=H5Fopen(files[i])
  h5dump(h,load=FALSE)
  raw_count=h$X
  sctype= h$Y
  
  c_names <- colnames(raw_count, do.NULL = FALSE)
  rownames <- rownames(raw_count, do.NULL = FALSE)
  colnames(raw_count) <- c_names
  rownames(raw_count) <- r_names

  adata.data <- raw_count
  adata <- CreateSeuratObject(counts=adata.data)
  
  adata <- NormalizeData(adata)
  adata <- FindVariableFeatures(adata, verbose = FALSE)
  adata <- ScaleData(adata)

  adata <- RunPCA(adata, features = VariableFeatures(object = adata))  
  adata <- FindNeighbors(adata)
  adata <- FindClusters(adata, random.seed =0)
  
  cluster_l = adata@meta.data[["seurat_clusters"]]
  cluster_l = as.numeric(cluster_l)
  cluster = as.matrix(cluster_l, ncol=1)
  
  ari_all[i] = ARI(c(cluster_l), c(sctype))
  nmi_all[i] = NMI(c(cluster_l), c(sctype))
  K[i] = length(unique(cluster_l))
}

results = data.frame(ari_all, nmi_all, K)
write.csv(results, "results/Seurat_with_sample.csv")
