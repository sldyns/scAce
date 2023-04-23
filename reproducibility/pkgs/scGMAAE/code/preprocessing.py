import scanpy as sc


def preprocessing(count):
    adata = sc.AnnData(count)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    adata = adata[:, adata.var.highly_variable]
    return adata
