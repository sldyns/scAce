rm(list=ls())

############################### library ###############################

library(Seurat)
library(rhdf5)
library(aricode)  # for calculating ARI and NMI


########################### Run without sampling ##############################

h=H5Fopen("../data/Human1.h5")
h5dump(h,load=FALSE)
raw_count=h$X
sctype= h$Y

colnames <- colnames(raw_count, do.NULL = FALSE)
rownames <- rownames(raw_count, do.NULL = FALSE)
colnames(raw_count) <- colnames
rownames(raw_count) <- rownames

adata.data <- raw_count
adata <- CreateSeuratObject(counts=adata.data, min.cells=3, min.features=200)
adata <- NormalizeData(adata)
adata <- FindVariableFeatures(adata)
adata <- ScaleData(adata)
adata <- RunPCA(adata, features = VariableFeatures(object = adata))  
adata <- FindNeighbors(adata)
adata <- FindClusters(adata, random.seed =0)
adata <- RunTSNE(adata, seed.use = 0)

cluster_l = adata@meta.data[["seurat_clusters"]]
cluster_l = as.numeric(cluster_l)
cluster = as.matrix(cluster_l, ncol=1)

ARI = ARI(c(cluster_l), c(sctype))
NMI = NMI(c(cluster_l), c(sctype))
K = length(unique(cluster_l))
TSNE = adata@reductions[["tsne"]]@cell.embeddings

results = data.frame(ARI, NMI, K, cluster, TSNE)
write.csv(results, "./results/Seurat_wo_sample.csv")




##################### Run 10 rounds with sampling 95% data ####################

ARI = c()
NMI = c()
K = c()

directory<-"../data/sample/data_Human1"
files <- list.files(directory, pattern = "^Human1_.*\\.h5$", full.names = TRUE)

for(i in 1:10){
  
  h=H5Fopen(files[i])
  h5dump(h,load=FALSE)
  raw_count=h$X
  sctype= h$Y
  
  if(i == 1){
    clusters = matrix(NA, nrow = dim(raw_count)[2], ncol=10, byrow = FALSE)
    labels = matrix(NA, nrow = dim(raw_count)[2], ncol=10, byrow = FALSE)
  }

  colnames <- colnames(raw_count, do.NULL = FALSE)
  rownames <- rownames(raw_count, do.NULL = FALSE)
  colnames(raw_count) <- colnames
  rownames(raw_count) <- rownames

  adata.data <- raw_count
  adata <- CreateSeuratObject(counts=adata.data, min.cells=3, min.features=200)
  
  adata <- NormalizeData(adata)
  adata <- FindVariableFeatures(adata)
  adata <- ScaleData(adata)

  adata <- RunPCA(adata, features = VariableFeatures(object = adata))  
  adata <- FindNeighbors(adata, dims = 1:10)
  adata <- FindClusters(adata, random.seed =0)
  
  cluster_l = adata@meta.data[["seurat_clusters"]]
  cluster_l = as.numeric(cluster_l)
  cluster = as.matrix(cluster_l, ncol=1)
  
  ARI[i] = ARI(c(cluster_l), c(sctype))
  NMI[i] = NMI(c(cluster_l), c(sctype))
  K[i] = length(unique(cluster_l))
  
  clusters[,i] = cluster_l
  labels[,i] = sctype
}

results_list = list(ARI, NMI, K, clusters, labels)
results = do.call(cbind, lapply(lapply(results_list, unlist), `length<-`, max(lengths(results_list))))
colnames(results) = c('ARI', 'NMI', 'K', rep('clusters', 10), rep('labels', 10))

write.csv(results, "./results/Seurat_with_sample.csv")

