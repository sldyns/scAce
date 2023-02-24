import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp

class scDataset(Dataset):
    def __init__(self, raw_mat, exp_mat, scale_factor):
        super(scDataset).__init__()
        if sp.issparse(raw_mat):
            raw_mat = raw_mat.todense()

        self.raw_mat = raw_mat.astype(np.float32)
        self.exp_mat = exp_mat.astype(np.float32)
        self.scale_factor = scale_factor.astype(np.float32)

    def __len__(self):
        return len(self.scale_factor)

    def __getitem__(self, idx):
        return idx, self.raw_mat[idx, :], self.exp_mat[idx, :], self.scale_factor[idx, :]

