rm(list=ls())

############################### library ###############################

library(rhdf5)
library(cidr)
library(Rtsne)
library(aricode)


########################### Run without sampling ##############################

h=H5Fopen("../data/Mouse2.h5")
h5dump(h,load=FALSE)
x = h$X
y = as.matrix(h$Y, ncol=1)

data <- scDataConstructor(as.matrix(x))
data <- determineDropoutCandidates(data)
data <- wThreshold(data)
data <- scDissim(data)
data <- scPCA(data, plotPC=FALSE)
data <- nPC(data)
nCluster(data)
data <- scCluster(data)

pc = data@PC
embedding = pc[,1:4]
tsne_out <- Rtsne(embedding, pca=FALSE, theta=0.0)
tsnes=tsne_out$Y
colnames(tsnes) <- c("tSNE_1", "tSNE_2")

ARI = ARI(c(data@clusters), c(y))
NMI = NMI(c(data@clusters), c(y))
K = data@nCluster
cluster = data@clusters

results = data.frame(ARI, NMI, K, cluster, tsnes, embedding)
write.csv(results, "./results/CIDR_wo_sample.csv")




##################### Run 10 rounds with sampling 95% data ####################

ARI = c()
NMI = c()
K = c()

directory<-"../data/sample/data_Mouse2"
files <- list.files(directory, pattern = "^Mouse2_.*\\.h5$", full.names = TRUE)

for(i in 1:10){
  h=H5Fopen(files[i])
  h5dump(h,load=FALSE)
  x = h$X
  y = as.matrix(h$Y, ncol=1)
  
  if(i == 1){
    clusters = matrix(NA, nrow = dim(x)[2], ncol=10, byrow = FALSE)
    labels = matrix(NA, nrow = dim(x)[2], ncol=10, byrow = FALSE)
  }

  data <- scDataConstructor(as.matrix(x))
  data <- determineDropoutCandidates(data)
  data <- wThreshold(data)
  data <- scDissim(data)
  data <- scPCA(data, plotPC=FALSE)
  data <- nPC(data)
  nCluster(data)
  data <- scCluster(data)
  
  ARI[i] = ARI(c(data@clusters), c(y))
  NMI[i] = NMI(c(data@clusters), c(y))
  K[i] = data@nCluster
  
  clusters[,i] = data@clusters
  labels[,i] = y
}

results_list = list(ARI, NMI, K, clusters, labels)
results = do.call(cbind, lapply(lapply(results_list, unlist), `length<-`, max(lengths(results_list))))
colnames(results) = c('ARI', 'NMI', 'K', rep('clusters', 10), rep('labels', 10))

write.csv(results, "./results/CIDR_with_sample.csv")