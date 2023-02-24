rm(list=ls())

############################### library ###############################

library(rhdf5)
library(cidr)
library(Rtsne)
library(aricode)


########################### Run without sampling ##############################

h=H5Fopen("../data/Sim.h5")
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

directory<-"../data/sample/data_Sim"
files <- list.files(directory, pattern = "^Sim_.*\\.h5$", full.names = TRUE)

for(i in 1:10){
  h=H5Fopen(files[i])
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
  
  ARI[i] = ARI(c(data@clusters), c(y))
  NMI[i] = NMI(c(data@clusters), c(y))
  K[i] = data@nCluster

}

results = data.frame(ARI, NMI, K)
write.csv(results, "./results/CIDR_with_sample.csv")