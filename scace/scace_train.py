import scanpy as sc
import numpy as np
import torch
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import scAce, centroid_merge, merge_compute
from .util import scDataset, ZINBLoss, ClusterLoss, ELOBkldLoss, clustering, calculate_metric, compute_mu


def run_scace(adata: sc.AnnData,
              n_epochs_pre: int = 300,
              n_epochs: int = 500,
              batch_size: int = 256,
              lr: float = 1e-4,
              resolution: float = 2,
              init_cluster=None,
              cl_type=None,
              save_pretrain: bool = False,
              saved_ckpt: str = None,
              pretrained_ckpt: str = None,
              return_all: bool = False):
    """
        Train scAce.
        Parameters
        ----------
        adata
            AnnData object of scanpy package.
        n_epochs_pre
            Number of total epochs in pre-training.
        n_epochs
            Number of total epochs in training.
        batch_size
            Number of cells for training in one epoch.
        lr
            Learning rate for AdamOptimizer.
        resolution
            The resolution parameter of sc.tl.leiden for the initial clustering.
        init_cluster
            Initial cluster results. If provided, perform cluster splitting after pre-training.
        save_pretrain
            If True, save the pre-trained model.
        saved_ckpt
            File name of pre-trained model to be saved, only used when save_pretrain is True.
        pretrained_ckpt
            File name of saved pre-trained model. If provided, load the saved pre-trained model without performing
            pre-training step.
        cl_type
            Cell type information. If provided, calculate ARI and NMI after clustering.
        return_all
            If True, print and return all temporary results.

        Returns
        -------
        adata
            AnnData object of scanpy package. Embedding and clustering result will be stored in adata.obsm['scace_emb']
            and adata.obs['scace_cluster']
        nmi
            Final NMI. Will be returned if 'return_all' is True and cell type information is provided.
        ari
            Final ARI. Will be returned if 'return_all' is True and cell type information is provided.
        K
            Final number of clusters. Will be returned if 'return_all' is True.
        pred_all
            All temporary clustering results. Will be returned if 'return_all' is True.
        emb_all
            All temporary embedding. Will be returned if 'return_all' is True.
    """

    ####################   Assert several input variables  ########################
    # To print and store all temporary results
    if return_all:
        pred_all, emb_all = [], []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################   Prepare data for training   ####################
    # Prepare data
    raw_mat, exp_mat = adata.raw.X, adata.X
    cell_type = adata.obs[cl_type] if cl_type is not None else None

    # Assume that 'scale_factor' has been calculated
    if 'scale_factor' not in adata.obs:
        scale_factor = np.ones((exp_mat.shape[0], 1), dtype=np.float32)
    else:
        scale_factor = adata.obs['scale_factor'].values.reshape(-1, 1)

    # Create dataset
    train_dataset = scDataset(raw_mat, exp_mat, scale_factor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    n_iters = len(train_loader)

    ####################   Set some parameters   #################
    # hyper-parameter
    kld_w = 0.001
    z_dim = 32
    encode_layers = [512]
    decode_layers = [512]
    activation = 'relu'

    # parameter for training
    tol, clu_w, m_numbers = 0.05, 1., 0
    merge_flag = True

    #######################   Prepare models & optimzers & loss  #######################
    input_dim = adata.X.shape[1]

    scace_model = scAce(input_dim=input_dim, device=device, z_dim=z_dim, encode_layers=encode_layers,
                        decode_layers=decode_layers, activation=activation).to(device)

    optimizer = optim.Adam(params=scace_model.parameters(), lr=lr)

    ZINB_Loss, KLD_Loss, Cluster_Loss = ZINBLoss(ridge_lambda=0), ELOBkldLoss(), ClusterLoss()

    ###########################   Pre-training   #########################
    if pretrained_ckpt:
        print('Pre-trained model provided, load checkpoint from file "{}".'.format(pretrained_ckpt))

        scace_model.load_state_dict(torch.load(pretrained_ckpt))

    else:
        print('Start pre-training! Total epochs is {}.'.format(n_epochs_pre))

        scace_model.pretrain = True

        # Start pre-training
        for epoch in tqdm(range(n_epochs_pre), unit='epoch', desc='Pre-training:'):
            avg_zinb, avg_kld, avg_loss = 0., 0., 0.

            for idx, raw, exp, sf in train_loader:
                raw, exp, sf = raw.to(device), exp.to(device), sf.to(device)
                z_mu, z_logvar, mu, disp, pi = scace_model(exp)

                # VAE Loss
                zinb_loss = ZINB_Loss(x=raw, mean=mu, disp=disp, pi=pi, scale_factor=sf)
                kld_loss = KLD_Loss(z_mu, z_logvar)

                loss = zinb_loss + kld_w * kld_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record losses
                if return_all:
                    avg_zinb += zinb_loss.item() / n_iters
                    avg_kld += kld_loss.item() / n_iters
                    avg_loss += loss.item() / n_iters

            if return_all:
                print('Pre-training epoch [{}/{}]. Average ZINB loss:{:.4f}, kld loss:{:.4f}, total loss:{:.4f}'
                      .format(epoch + 1, n_epochs_pre, avg_zinb, avg_kld, avg_loss))

        # Finish pre-training
        scace_model.pretrain = False

        print('Finish pre-training!')

        if save_pretrain:
            torch.save(scace_model.state_dict(), saved_ckpt)

    ###########################   Find initial clustering centers  #########################

    # Initial clustering
    if init_cluster is not None:
        print('Perform initial clustering through cluster split with provided cluster labels')
        y_pred_last, mu, scace_emb = clustering(scace_model, exp_mat, init_cluster=init_cluster)

    else:
        print('Perform initial clustering through Leiden with resolution = {}'.format(resolution))
        y_pred_last, mu, scace_emb = clustering(scace_model, exp_mat, resolution=resolution)

    # Number of initial clusters
    n_clusters = len(np.unique(y_pred_last))
    print('Finish initial clustering! Number of initial clusters is {}'.format(n_clusters))

    # Initial parameter mu
    scace_model.mu = Parameter(torch.Tensor(n_clusters, scace_model.z_dim).to(device))
    optimizer = optim.Adam(params=scace_model.parameters(), lr=lr)
    scace_model.mu.data.copy_(torch.Tensor(mu))

    # Store initial tsne plot and clustering result
    if return_all:
        emb_all.append(scace_emb)
        pred_all.append(y_pred_last)

        # If there provide ground truth cell type information, calculate NMI and ARI
        if cl_type is not None:
            nmi, ari = calculate_metric(y_pred_last, cell_type)
            print('Initial Clustering: NMI= %.4f, ARI= %.4f' % (nmi, ari))

    ############################   Training   #########################

    print('Start training! Total epochs is {}.'.format(n_epochs))

    # Calculate q, p firstly
    y_pred, scace_emb, q, p = clustering(scace_model, exp_mat)

    # Start training
    for epoch in tqdm(range(n_epochs), unit='epoch', desc='Training:'):

        # Check stop & cluster merging criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / len(y_pred)
        y_pred_last = y_pred

        if epoch > 0 and delta_label < tol:
            if not merge_flag: print("Reach tolerance threshold. Stopping training."); break

            print('Reach tolerance threshold. Perform cluster merging.')

            mu_prepare = scace_model.mu.cpu().detach().numpy()
            y_pred, Centroid, d_bar, intra_dis, d_ave, n_clusters = merge_compute(y_pred, mu_prepare, scace_emb)

            Final_Centroid_merge, Label_merge, n_clusters_t, pred_t = centroid_merge(scace_emb, Centroid, y_pred, d_bar, intra_dis, d_ave)
            m_numbers += 1

            # If n_clusters not change, stop merging clusters
            if m_numbers > 1 and n_clusters_t == n_clusters:
                merge_flag, tol = False, tol / 10.
                print('Stop merging clusters! Continue updating several rounds.')

            else:
                n_clusters = n_clusters_t

                y_pred = Label_merge

                mu = compute_mu(scace_emb, y_pred)

                scace_model.mu = Parameter(torch.Tensor(n_clusters, scace_model.z_dim).to(device))
                optimizer = optim.Adam(params=scace_model.parameters(), lr=lr)
                scace_model.mu.data.copy_(torch.Tensor(mu))

                q = scace_model.soft_assign(torch.tensor(scace_emb).to(device))
                p = scace_model.target_distribution(q)

            # Store tsne plot and clustering results of each cluster merging
            if return_all:
                emb_all.append(scace_emb)
                pred_all.append(pred_t)

        # Start training
        avg_zinb, avg_kld, avg_clu, avg_loss = 0., 0., 0., 0.

        for idx, raw, exp, sf in train_loader:
            raw, exp, sf = raw.to(device), exp.to(device), sf.to(device)

            z_mu, z_logvar, mu, disp, pi, q = scace_model(exp)

            # VAE Losses
            zinb_loss = ZINB_Loss(x=raw, mean=mu, disp=disp, pi=pi, scale_factor=sf)
            kld_loss = KLD_Loss(z_mu, z_logvar)

            # DEC Loss
            clu_loss = Cluster_Loss(p[idx].detach(), q)

            # All losses
            loss = zinb_loss + kld_w * kld_loss + clu_w * clu_loss

            # Optimize VAE + DEC
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            if return_all:
                avg_zinb += zinb_loss.item() / n_iters
                avg_kld += kld_loss.item() / n_iters
                avg_clu += clu_loss.item() / n_iters
                avg_loss += loss.item() / n_iters

        if return_all:
            print(
                'Train epoch [{}/{}]. ZINB loss:{:.4f}, kld loss:{:.4f}, cluster loss:{:.4f}, total loss:{:.4f}'.format(
                    epoch + 1, n_epochs, avg_zinb, avg_kld, avg_clu, avg_loss))

        # Update the targe distribution p
        y_pred, scace_emb, q, p = clustering(scace_model, exp_mat)

        if cl_type is not None:
            nmi, ari = calculate_metric(y_pred, cell_type)
            print('Clustering   %d: NMI= %.4f, ARI= %.4f, Delta=%.4f' % (
                epoch + 1, nmi, ari, delta_label))

    ############################   Return results   #########################
    adata.obsm['scace_emb'] = scace_emb
    adata.obs['scace_cluster'] = y_pred

    K = len(np.unique(y_pred))

    if return_all:
        if cl_type is not None:
            return adata, nmi, ari, K, pred_all, emb_all
        return adata, K, pred_all, emb_all

    return adata
