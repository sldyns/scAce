import h5py
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils import data
from model.GMAAE import *
from preprocessing import preprocessing
from sklearn.mixture import GaussianMixture
from evaluation import eva


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
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if args.cuda:
  torch.cuda.manual_seed(SEED)


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


f = h5py.File('data/baron-human.h5')
adata = preprocessing(np.array(f['X']))
adata.obs['Group'] = np.array(f['Y'])


train_dataset = MyDataset(adata=adata,
                          transform=transforms.Compose([
                           ToTensor()
                          ]))

args.batch_size = 128
args.input_size = 1000


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


print(args.input_size)
#########################################################
## Train and Test Model
#########################################################
args.num_classes = 14
args.gaussian_size = 32
args.epochs = 30
gmaae = GMAAE(args)

## Training Phase
Q, P = gmaae.generate_model(train_loader)

embedding, true_labels = gmaae.create_latent(Q, train_loader)
true_labels = true_labels.astype(np.int8)


# clustering
GM = GaussianMixture(n_components=14)
pred_label = GM.fit_predict(embedding)
acc, nmi, ari = eva(true_labels, pred_label)

GM = GaussianMixture(n_components=14)
pred_label = GM.fit_predict(embedding)
acc, nmi, ari = eva(true_labels, pred_label)

GM = GaussianMixture(n_components=14)
pred_label = GM.fit_predict(embedding)
acc, nmi, ari = eva(true_labels, pred_label)

GM = GaussianMixture(n_components=14)
pred_label = GM.fit_predict(embedding)
acc, nmi, ari = eva(true_labels, pred_label)

GM = GaussianMixture(n_components=14)
pred_label = GM.fit_predict(embedding)
acc, nmi, ari = eva(true_labels, pred_label)
