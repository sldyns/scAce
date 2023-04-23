rm(list=ls())

############################### library ###############################

library(rhdf5)
library(cidr)
library(aricode)


########################### Run without sampling ##############################

h=H5Fopen("../data/Mouse_E.h5")
h5dump(h,load=FALSE)
x = h$X
y = as.matrix(h$Y, ncol=1)

start_time <- proc.time()
data <- scDataConstructor(as.matrix(x))
data <- determineDropoutCandidates(data)
data <- wThreshold(data)
data <- scDissim(data)
data <- scPCA(data, plotPC=FALSE)
data <- nPC(data)
nCluster(data)
data <- scCluster(data)

end_time <- proc.time()
Time_use <- end_time - start_time

pc = data@PC
Embedding = pc[,1:4]

ari = ARI(c(data@clusters), c(y))
nmi = NMI(c(data@clusters), c(y))
K = data@nCluster
Cluster = data@clusters

results = data.frame(ari, nmi, K, Cluster, Embedding, Time_use[3])
write.csv(results, "results/CIDR_wo_sample.csv")



##################### Run 10 rounds with sampling 95% data ####################

ari_all = c()
nmi_all = c()
K = c()

directory<-"../data/sample/Mouse_E"
files <- list.files(directory, pattern = "^Mouse_E_.*\\.h5$", full.names = TRUE)

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
  
  ari_all[i] = ARI(c(data@clusters), c(y))
  nmi_all[i] = NMI(c(data@clusters), c(y))
  K[i] = data@nCluster

  clusters[,i] = data@clusters
  labels[,i] = y

}

results_list = list(ari_all, nmi_all, K, clusters, labels)
results = do.call(cbind, lapply(lapply(results_list, unlist), `length<-`, max(lengths(results_list))))
colnames(results) = c('ARI', 'NMI', 'K', rep('clusters', 10), rep('labels', 10))

write.csv(results, "results/CIDR_with_sample.csv")