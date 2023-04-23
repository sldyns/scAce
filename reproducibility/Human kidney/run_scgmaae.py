import argparse
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from GMAAE import *
from sklearn.mixture import GaussianMixture
from GMAAE.evaluation import eva
import time
import h5py

from reproducibility.utils import data_sample, data_preprocess, set_seed, read_data, preprocessing_scgmaae



parser = argparse.ArgumentParser(description='PyTorch Implementation of scGMAAE')

## Used only in notebooks
parser.add_argument('-f', '--file',
                    help='Path for input file. First line should contain number of lines to search in')

## Dataset

parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

## GPU
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')

## Training
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=200, type=int,
                    help='mini-batch size of validation (default: 200)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay_epoch', default=-1, type=int,
                    help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')

## Architecture
parser.add_argument('--num_classes', type=int, default=10,
                    help='number of classes (default: 10)')
parser.add_argument('--gaussian_size', default=32, type=int,
                    help='gaussian size (default: 32)')
parser.add_argument('--input_size', default=1000, type=int,
                    help='input size (default: 1000)')

## Partition parameters
parser.add_argument('--train_proportion', default=1.0, type=float,
                    help='proportion of examples to consider for training only (default: 1.0)')

## Gumbel parameters
parser.add_argument('--init_temp', default=1.0, type=float,
                    help='Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)')
parser.add_argument('--decay_temp', default=1, type=int,
                    help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
parser.add_argument('--hard_gumbel', default=0, type=int,
                    help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
parser.add_argument('--min_temp', default=0.5, type=float,
                    help='Minimum temperature of gumbel-softmax after annealing (default: 0.5)' )
parser.add_argument('--decay_temp_rate', default=0.013862944, type=float,
                    help='Temperature decay rate at every epoch (default: 0.013862944)')

## Loss function parameters
parser.add_argument('--w_gauss', default=1, type=float,
                    help='weight of gaussian loss (default: 1)')
parser.add_argument('--w_categ', default=1, type=float,
                    help='weight of categorical loss (default: 1)')
parser.add_argument('--w_rec', default=1, type=float,
                    help='weight of reconstruction loss (default: 1)')
parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                    default='bce', help='desired reconstruction loss function (default: bce)')

## Others
parser.add_argument('--verbose', default=0, type=int,
                    help='print extra information at every epoch.(default: 0)')
parser.add_argument('--random_search_it', type=int, default=20,
                    help='iterations of random search (default: 20)')

args = parser.parse_args(args=[])

if args.cuda == 1:
   os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

## Random Seed
set_seed(0)


class MyDataset():
    """Operations with the datasets."""

    def __init__(self, adata, transform=None):
        self.data = adata.X
        self.data_cls = np.array(adata.obs['Group'])
        self.transform = transform

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.data_cls[idx]
        sample = (data, label)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample[0], sample[1]
        return (torch.from_numpy(data), torch.from_numpy(label))


####################################  Read dataset  ####################################

data_mat = h5py.File('../data/Human_k.h5')
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])


####################################  Run without sampling  ####################################

adata = preprocessing_scgmaae(x)
adata.obs['Group'] = np.array(y)


train_dataset = MyDataset(adata=adata,
                          transform=transforms.Compose([
                           ToTensor()
                          ]))

args.input_size = 1000
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print(args.input_size)

start = time.time()

gmaae = GMAAE(args)

## Training Phase
Q, P = gmaae.generate_model(train_loader)

embedding, true_labels = gmaae.create_latent(Q, train_loader)
true_labels = true_labels.astype(np.int8)

# clustering
GM = GaussianMixture(n_components=11)
pred_label = GM.fit_predict(embedding)

end = time.time()
run_time = end - start
print(f'Total time: {end - start} seconds')

acc, nmi, ari = eva(true_labels, pred_label)

print("ARI: ", ari)
print("NMI:", nmi)
np.savez("results/scGMAAE_wo_sample_sim.npz", ARI=ari, NMI=nmi, Clusters=pred_label,
         Labels=true_labels, Embedding=embedding, Time_use=run_time)



####################################  Run 10 rounds with sampling 95% data  ####################################

total_rounds = 10
ari_all, nmi_all = [], []

for i in range(total_rounds):

    print('----------------Round: %d-------------------' % int(i + 1))
    seed = 10 * i
    set_seed(0)
    x_sample, y_sample = data_sample(x, y, seed)

    set_seed(0)

    adata = preprocessing_scgmaae(x_sample)
    adata.obs['Group'] = np.array(y_sample)

    train_dataset = MyDataset(adata=adata,
                              transform=transforms.Compose([
                               ToTensor()
                              ]))

    args.input_size = 1000

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    gmaae = GMAAE(args)

    Q, P = gmaae.generate_model(train_loader)

    embedding, true_labels = gmaae.create_latent(Q, train_loader)
    true_labels = true_labels.astype(np.int8)

    # clustering
    GM = GaussianMixture(n_components=11)
    pred_label = GM.fit_predict(embedding)

    acc, nmi, ari = eva(true_labels, pred_label)
    nmi_all.append(nmi)
    ari_all.append(ari)
    print(ari)

print(ari_all)
print(nmi_all)

np.savez("results/scGMAAE_with_sample.npz", ARI=ari_all, NMI=nmi_all)