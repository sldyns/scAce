## Summary of the simulated data

For simulated data, we used the R package scDesign2 to generate a synthetic single-cell gene expression matrix
with ground truth cell type labels. ln this simulated dataset, there were 16,653 genes and 1150 cells belonging to five
cell types. The number of cells in each cell type was 600, 200, 200, 100, and 50, respectively. The last cell type
accounted for less than 5% of the cells and was used to represent a rare cell type. The real scRNA-seg dataset used
by scDesign2 to learn gene expression parameters was a peripheral blood mononuclear cell (PBMC) dataset generated
by the 10x Genomics technology (PBMC dataset can be downloaded from 
https://github.com/tgump/scDeepCluster/blob/master/scRNA-seg%20data/10X_PBMC.h5).

## Summary of the all scRNA-seq datasets

|      Datasets      | No. of cells | No. of genes | No. of cell types |     Source      |
| :----------------: | :----------: | :----------: | :---------------: | :-------------: |
|   Human pancreas   |     3605     |    20125     |        14         |   GSM2230759    |
|     Human PBMC     |     4271     |    16653     |         8         |    GSE96583     |
|    Human kidney    |     5685     |    25125     |        11         | EGAS00001002171 |
|      Mouse ES      |     2717     |    24047     |         4         |    GSE65525     |
| Mouse hypothalamus |    12089     |    23284     |        46         |    GSE94333     |
|    Mouse kidney    |     2717     |    23797     |         8         |    GSE65525     |
|    Turtle brain    |    18664     |    23500     |        15         |   PRJNA408230   |
|  Simulation data   |     1150     |    16653     |         5         |        -        |


## Download

All datasets can be downloaded from https://drive.google.com/drive/folders/1c33An3HNdJQhazoy_ky9E-lCc3a4y7fl

or downloaded from the following links.

|      Dataset       |                 Website                  |
| :----------------: | :--------------------------------------: |
|   Human pancreas   | <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM22307659 |
|     Human PBMC     | <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583 |
|    Human kidney    | https://ega-archive.org/studies/EGAS00001002171 |
|      Mouse ES      | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525 |
| Mouse hypothalamus | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87544 |
|    Mouse kidney    | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE95333 |
|    Turtle brain    | https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA408230 |

After download these datasets, please run `sample_datasets.py` to generate the sampled datasets for reproduciblity.
