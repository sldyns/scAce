import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from networks.Networks import *
from networks.Layers import Gaussian
from losses.LossFunctions import *
from metrics.Metrics import *
from evaluation import eva

class GMAAE:

  def __init__(self, args):
    self.num_epochs = args.epochs
    self.cuda = args.cuda
    self.verbose = args.verbose

    self.batch_size = args.batch_size
    self.batch_size_val = args.batch_size_val
    self.learning_rate = args.learning_rate
    self.decay_epoch = args.decay_epoch
    self.lr_decay = args.lr_decay
    self.w_cat = args.w_categ
    self.w_gauss = args.w_gauss    
    self.w_rec = args.w_rec
    self.rec_type = args.rec_type 

    self.num_classes = args.num_classes
    self.gaussian_size = args.gaussian_size
    self.input_size = args.input_size

    # gumbel
    self.init_temp = args.init_temp
    self.decay_temp = args.decay_temp
    self.hard_gumbel = args.hard_gumbel
    self.min_temp = args.min_temp
    self.decay_temp_rate = args.decay_temp_rate
    self.gumbel_temp = self.init_temp

    self.InferenceNet = InferenceNet(self.input_size, self.gaussian_size, self.num_classes)
    self.GenerativeNet = GenerativeNet(self.input_size, self.gaussian_size, self.num_classes)
    self.D_net_gauss = D_net_gauss(self.gaussian_size)
    # self.network = GMVAENet(self.input_size, self.gaussian_size, self.num_classes)
    self.losses = LossFunctions()
    self.metrics = Metrics()

    if self.cuda:
      self.InferenceNet = self.InferenceNet.cuda()
      self.GenerativeNet = self.GenerativeNet.cuda()
      self.D_net_gauss =  self.D_net_gauss.cuda()
      # self.network = self.network.cuda()


  def train_epoch(self, P_decoder, Q_encoder, Q_generator, D_gauss_solver, data_loader):
    '''
    Train procedure for one epoch.
    '''
    # Set the networks in train mode (apply dropout when needed)
    self.InferenceNet.train()
    self.GenerativeNet.train()
    self.D_net_gauss.train()

    # The batch size has to be a divisor of the size of the dataset or it will return
    # invalid samples
    for X, target in data_loader:
      # Load batch and normalize samples to be between 0 and 1
      X, target = Variable(X), Variable(target)
      if self.cuda:
        X, target = X.cuda(), target.cuda()

      # Init gradients
      self.InferenceNet.zero_grad()
      self.GenerativeNet.zero_grad()
      self.D_net_gauss.zero_grad()
      #######################
      # Reconstruction phase
      #######################
      out_inf = self.InferenceNet(X, self.gumbel_temp, self.hard_gumbel)
      z_sample, y = out_inf['gaussian'], out_inf['categorical']
      out_gen = self.GenerativeNet(z_sample, y)
      X_sample = out_gen['x_rec']

      recon_loss = self.losses.reconstruction_loss(real=X, predicted=X_sample, rec_type=self.rec_type)
      recon_loss.backward()
      P_decoder.step()
      Q_encoder.step()

      self.InferenceNet.zero_grad()
      self.GenerativeNet.zero_grad()
      self.D_net_gauss.zero_grad()

      #######################
      # Regularization phase
      #######################
      # Discriminator
      self.InferenceNet.eval()
      out_inf = self.InferenceNet(X, self.gumbel_temp, self.hard_gumbel)
      z_sample, y = out_inf['gaussian'], out_inf['categorical']
      out_gen = self.GenerativeNet(z_sample, y)
      y_mu, y_var = out_gen['y_mean'], out_gen['y_var']
      z_real_gauss = y_mu
      if self.cuda:
        z_real_gauss = z_real_gauss.cuda()

      z_fake_gauss = out_inf['mean']

      D_real_gauss = self.D_net_gauss(z_real_gauss)
      D_fake_gauss = self.D_net_gauss(z_fake_gauss)
      D_loss = self.losses.D_loss(D_real_gauss, D_fake_gauss)

      D_loss.backward()
      D_gauss_solver.step()

      self.InferenceNet.zero_grad()
      self.GenerativeNet.zero_grad()
      self.D_net_gauss.zero_grad()

      # Generator
      self.InferenceNet.train()
      out_inf = self.InferenceNet(X, self.gumbel_temp, self.hard_gumbel)
      z_fake_gauss = out_inf['mean']

      D_fake_gauss = self.D_net_gauss(z_fake_gauss)
      G_loss = self.losses.G_loss(D_fake_gauss)

      G_loss.backward()
      Q_generator.step()

      self.InferenceNet.zero_grad()
      self.GenerativeNet.zero_grad()
      self.D_net_gauss.zero_grad()

    return D_loss, G_loss, recon_loss


  def generate_model(self, data_loader):
    torch.manual_seed(10)

    if self.cuda:
      Q = self.InferenceNet.cuda()
      P = self.GenerativeNet.cuda()
      D_gauss = self.D_net_gauss.cuda()
    else:
      Q = self.InferenceNet
      P = self.GenerativeNet
      D_gauss = self.D_net_gauss

    # Set learning rates
    # gen_lr = 0.0001
    # reg_lr = 0.00005
    gen_lr = 0.001
    reg_lr = 0.0005

    # Set optimizators
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

    for epoch in range(1, self.num_epochs + 1):
      D_loss_gauss, G_loss, recon_loss = self.train_epoch(P_decoder, Q_encoder, Q_generator,
                                                                         D_gauss_solver, data_loader)
      if epoch % 10 == 0:
        print('Epoch-{}; D_loss_gauss: {:.4}; G_loss: {:.4}; recon_loss: {:.4};'
              .format(epoch, D_loss_gauss.data, G_loss.data, recon_loss.data))

    return Q, P


  def create_latent(self, Q, loader):
    '''
    Creates the latent representation for the samples in loader
    return:
        z_values: numpy array with the latent representations
        labels: the labels corresponding to the latent representations
    '''
    Q.eval()
    labels = []

    for batch_idx, (X, target) in enumerate(loader):
      X, target = Variable(X), Variable(target)
      labels.extend(target.data.tolist())
      if self.cuda:
        X, target = X.cuda(), target.cuda()
      z_sample = Q(X)['mean']
      if batch_idx > 0:
        z_values = np.concatenate((z_values, np.array(z_sample.data.tolist())))
      else:
        z_values = np.array(z_sample.data.tolist())
    labels = np.array(labels).reshape(-1)

    return z_values, labels


  def create_recconstructed(self, P, Q, loader):
    '''
    Creates the latent representation for the samples in loader
    return:
        z_values: numpy array with the latent representations
        labels: the labels corresponding to the latent representations
    '''
    P.eval()
    Q.eval()

    for batch_idx, (X, target) in enumerate(loader):
      X, target = Variable(X), Variable(target)
      if self.cuda:
        X, target = X.cuda(), target.cuda()
      out_inf = Q(X, self.gumbel_temp, self.hard_gumbel)
      z_sample, y = out_inf['gaussian'], out_inf['categorical']
      out_gen = P(z_sample, y)
      x_sample = out_gen['x_rec']
      if batch_idx > 0:
        x_values = np.concatenate((x_values, np.array(x_sample.data.tolist())))
      else:
        x_values = np.array(x_sample.data.tolist())

    return x_values














