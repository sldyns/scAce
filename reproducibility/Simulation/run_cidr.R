rm(list=ls())

############################### library ###############################

library(rhdf5)
library(cidr)
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
Embedding = pc[,1:4]

ari = ARI(c(data@clusters), c(y))
nmi = NMI(c(data@clusters), c(y))
K = data@nCluster
Cluster = data@clusters

results = data.frame(ari, nmi, K, Cluster, Embedding)
write.csv(results, "results/CIDR_wo_sample.csv")



##################### Run 10 rounds with sampling 95% data ####################

ari_all = c()
nmi_all = c()
K = c()

directory<-"../data/sample/Sim"
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
  
  ari_all[i] = ARI(c(data@clusters), c(y))
  nmi_all[i] = NMI(c(data@clusters), c(y))
  K[i] = data@nCluster

}

results = data.frame(ari_all, nmi_all, K)
write.csv(results, "results/CIDR_with_sample.csv")