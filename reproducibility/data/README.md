## Summary of the simulated data

For simulated data, we used the R package scDesign2 to generate a synthetic single-cell gene expression matrix
with ground truth cell type labels. ln this simulated dataset, there were 16,653 genes and 1150 cells belonging to five
cell types. The number of cells in each cell type was 600, 200, 200, 100, and 50, respectively. The last cell type
accounted for less than 5% of the cells and was used to represent a rare cell type. The real scRNA-seg dataset used
by scDesign2 to learn gene expression parameters was a peripheral blood mononuclear cell (PBMC) dataset generated
by the 10x Genomics technology (PBMC dataset can be downloaded from 
https://github.com/tgump/scDeepCluster/blob/master/scRNA-seg%20data/10X_PBMC.h5).

## Summary of the all scRNA-seq datasets

|    Datasets     | No. of cells | No. of cell types |   Source   |
|:---------------:|:------------:|:-----------------:|:----------:|
|     Human1      |     1724     |        14         | GSM2230758 |
|     Human2      |     3605     |        14         | GSM2230759 |
|     Human3      |     1303     |        14         | GSM2230760 |
|     Mouse1      |     3660     |         8         |  GSE94333  |
|     Mouse2      |     3909     |         6         | GSE109774  |
|     Mouse3      |     2717     |         4         |  GSE65525  |
| Simulation data |     1150     |         5         |     -      |


## Download

All datasets can be downloaded from https://drive.google.com/drive/folders/1c33An3HNdJQhazoy_ky9E-lCc3a4y7fl

or downloaded from the following links.

| Dataset |                 Website                  |
| :-----: | :--------------------------------------: |
| Human1  | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2230758 |
| Human2  | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2230759 |
| Human3  | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2230760 |
| Mouse1  | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94333 |
| Mouse2  | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774 |
| Mouse3  | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525 |

After download these datasets, please run `sample_datasets.py` to generate the sampled datasets for reproduciblity.
