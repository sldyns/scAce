from reproducibility.utils import data_sample, read_data
import numpy as np
import h5py
import os

data_names = ['Human_k', 'Human_p', 'Human_PBMC', 'Mouse_E', 'Mouse_h', 'Mouse_k', 'Turtle_b', 'Sim']

for data_name in data_names:
    print('Sampling data: ' + data_name + ' ...')

    if data_name in ['Mouse_E', 'Mouse_h', 'Mouse_k', 'Turtle_b']:
        mat, obs, var, uns = read_data(data_name+'.h5', sparsify=False, skip_exprs=False)
        x = np.array(mat.toarray())
        cell_name = np.array(obs["cell_type1"])
        cell_type, y = np.unique(cell_name, return_inverse=True)

    else:
        data_mat = h5py.File(data_name+'.h5')
        x, y = np.array(data_mat['X']), np.array(data_mat['Y'])
        data_mat.close()

    path = './sample/' + data_name + '/'
    os.makedirs(path)

    for i in range(10):

        seed = 10 * i
        x_sample, y_sample = data_sample(x, y, seed)
        h5_path = path + data_name + '_' + str(i + 1)+'.h5'
        f = h5py.File(h5_path, "w")
        f.create_dataset("X", data=x_sample)
        f.create_dataset("Y", data=y_sample)
        f.close()