import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        layer = nn.Linear(layers[i - 1], layers[i])
        nn.init.kaiming_normal_(layer.weight)
        # nn.init.kaiming_normal_(layer.bias)
        nn.init.constant_(layer.bias, 0)
        net.append(layer)

        # net.append(nn.BatchNorm1D(layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "mish":
            net.append(Mish())
        elif activation == "tanh":
            net.append(nn.Tanh())
    return nn.Sequential(*net)


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x.shape)
        return x


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class scAce(nn.Module):
    def __init__(self, input_dim, device, z_dim=32,
                 encode_layers=[512, 256],
                 decode_layers=[256, 512],
                 activation='relu'):
        super(scAce, self).__init__()
        self.z_dim = z_dim
        self.activation = activation

        # self.mu = None
        self.pretrain = False
        self.device = device
        self.alpha = 1.
        self.sigma = 0  # 1.

        self.encoder = buildNetwork([input_dim] + encode_layers, activation=activation)
        self.decoder = buildNetwork([z_dim] + decode_layers, activation=activation)

        self.enc_mu = nn.Linear(encode_layers[-1], z_dim)
        self.enc_var = nn.Linear(encode_layers[-1], z_dim)

        self.dec_mean = nn.Sequential(nn.Linear(decode_layers[-1], input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(decode_layers[-1], input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(decode_layers[-1], input_dim), nn.Sigmoid())


    def soft_assign(self, z):

        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2))
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def reparameterize(self, mu, logvar):

        # obtain standard deviation from log variance
        std = torch.exp(0.5 * logvar)
        # values are sampled from unit normal distribution
        eps = torch.randn(std.shape).to(self.device)
        return mu + eps * std

    def Encoder(self, x):
        # h = self.encoder(x + torch.randn_like(x) * self.sigma)
        h = self.encoder(x)
        z_mu = self.enc_mu(h)
        z_logvar = self.enc_var(h)
        z = self.reparameterize(z_mu, z_logvar)

        if self.pretrain:
            return z_mu, z_logvar, z

        h0 = self.encoder(x)
        z_mu0 = self.enc_mu(h0)
        z_logvar0 = self.enc_var(h0)
        z0 = self.reparameterize(z_mu0, z_logvar0)
        return z_mu, z_logvar, z, z0

    def Decoder(self, z):

        h = self.decoder(z)
        mu = self.dec_mean(h)
        disp = self.dec_disp(h)
        pi = self.dec_pi(h)

        return mu, disp, pi

    def forward(self, x):
        if self.pretrain:
            # Encode
            z_mu, z_logvar, z = self.Encoder(x)

            # Decode
            mu, disp, pi = self.Decoder(z)

            return z_mu, z_logvar, mu, disp, pi

        # else

        # Encode
        z_mu, z_logvar, z, z0 = self.Encoder(x)

        # Decode
        mu, disp, pi = self.Decoder(z)

        # cluster
        q = self.soft_assign(z0)

        return z_mu, z_logvar, mu, disp, pi, q

    def EncodeAll(self, X, batch_size=256):
        all_z_mu = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            exp = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            exp = torch.tensor(np.float32(exp))
            with torch.no_grad():
                z_mu, _, _, _ = self.Encoder(exp.to(self.device))

            all_z_mu.append(z_mu)

        all_z_mu = torch.cat(all_z_mu, dim=0)
        return all_z_mu


def weight_init(m):
    nn.init.xavier_normal_(m.weight)
    # nn.init.kaiming_normal_(m.bias)
    nn.init.constant_(m.bias, 0)
