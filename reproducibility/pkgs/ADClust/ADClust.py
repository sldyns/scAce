import ctypes
import os

from model import _ADClust_Autoencoder
from utils import *

C_DIP_FILE = None


def dip_test(X, is_data_sorted=False, debug=False):
    n_points = X.shape[0]
    data_dip = dip(X, just_dip=True, is_data_sorted=is_data_sorted, debug=debug)
    pval = dip_pval(data_dip, n_points)
    return data_dip, pval


def dip(X, just_dip=False, is_data_sorted=False, debug=False):
    assert X.ndim == 1, "Data must be 1-dimensional for the dip-test. Your shape:{0}".format(X.shape)

    N = len(X)
    if not is_data_sorted:
        X = np.sort(X)
    if N < 4 or X[0] == X[-1]:
        d = 0.0
        return d if just_dip else (d, None, None)

    # Prepare data to match C data types
    if C_DIP_FILE is None:
        load_c_dip_file()
    X = np.asarray(X, dtype=np.float64)
    X_c = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    N_c = np.array([N]).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    dip_value = np.zeros(1, dtype=np.float).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    low_high = np.zeros(4).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    modal_triangle = np.zeros(3).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    gcm = np.zeros(N).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    lcm = np.zeros(N).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    mn = np.zeros(N).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    mj = np.zeros(N).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    debug_c = np.array([1 if debug else 0]).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Execute C dip test
    _ = C_DIP_FILE.diptst(X_c, N_c, dip_value, low_high, modal_triangle, gcm, lcm, mn, mj, debug_c)
    dip_value = dip_value[0]
    if just_dip:
        return dip_value
    else:
        low_high = (low_high[0], low_high[1], low_high[2], low_high[3])
        modal_triangle = (modal_triangle[0], modal_triangle[1], modal_triangle[2])
        return dip_value, low_high, modal_triangle


def load_c_dip_file():
    global C_DIP_FILE
    files_path = os.path.dirname(__file__)
    if platform.system() == "Windows":
        dip_compiled = files_path + "/dip.dll"
    else:
        dip_compiled = "dip.so"

    print(dip_compiled)
    if os.path.isfile(dip_compiled):
        # load c file
        try:

            C_DIP_FILE = ctypes.CDLL(dip_compiled)
            C_DIP_FILE.diptst.restype = None
            C_DIP_FILE.diptst.argtypes = [ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int)]
        except Exception as e:
            print("[WARNING] Error while loading the C compiled dip file.")
            raise e
    else:
        raise Exception("C compiled dip file can not be found.\n"
                        "On Linux execute: gcc -fPIC -shared -o dip.so dip.c\n"
                        "Or Please ensure the dip.so was added in your LD_LIBRARY_PATH correctly by executing \n"
                        "(export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./dip.so)   in the current directory of the ADClust folder. \n")


def _merge_by_dip_value(X, embedded_data, cluster_labels_cpu, dip_argmax, n_clusters_current, centers_cpu,
                        embedded_centers_cpu):
    # Get points in clusters
    points_in_center_1 = len(cluster_labels_cpu[cluster_labels_cpu == dip_argmax[0]])
    points_in_center_2 = len(cluster_labels_cpu[cluster_labels_cpu == dip_argmax[1]])

    # update labels
    for j, l in enumerate(cluster_labels_cpu):
        if l == dip_argmax[0] or l == dip_argmax[1]:
            cluster_labels_cpu[j] = n_clusters_current - 1
        elif l < dip_argmax[0] and l < dip_argmax[1]:
            cluster_labels_cpu[j] = l
        elif l > dip_argmax[0] and l > dip_argmax[1]:
            cluster_labels_cpu[j] = l - 2
        else:
            cluster_labels_cpu[j] = l - 1

    # Find new center position
    optimal_new_center = (embedded_centers_cpu[dip_argmax[0]] * points_in_center_1 +
                          embedded_centers_cpu[dip_argmax[1]] * points_in_center_2) / (
                                 points_in_center_1 + points_in_center_2)
    new_center_cpu, new_embedded_center_cpu = get_nearest_points_to_optimal_centers(X, [optimal_new_center],
                                                                                    embedded_data)
    # Remove the two old centers and add the new one
    centers_cpu_tmp = np.delete(centers_cpu, dip_argmax, axis=0)
    centers_cpu = np.append(centers_cpu_tmp, new_center_cpu, axis=0)
    embedded_centers_cpu_tmp = np.delete(embedded_centers_cpu, dip_argmax, axis=0)
    embedded_centers_cpu = np.append(embedded_centers_cpu_tmp, new_embedded_center_cpu, axis=0)

    # Update dip values
    dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_current)
    return cluster_labels_cpu, centers_cpu, embedded_centers_cpu, dip_matrix_cpu


def _get_dip_matrix(data, dip_centers, dip_labels, n_clusters, max_cluster_size_diff_factor=3, min_sample_size=100):
    dip_matrix = np.zeros((n_clusters, n_clusters))

    # Loop over all combinations of centers
    for i in range(0, n_clusters - 1):
        for j in range(i + 1, n_clusters):
            center_diff = dip_centers[i] - dip_centers[j]
            points_in_i = data[dip_labels == i]
            points_in_j = data[dip_labels == j]
            points_in_i_or_j = np.append(points_in_i, points_in_j, axis=0)
            proj_points = np.dot(points_in_i_or_j, center_diff)
            _, dip_p_value = dip_test(proj_points)

            # Check if clusters sizes differ heavily
            if points_in_i.shape[0] > points_in_j.shape[0] * max_cluster_size_diff_factor or \
                    points_in_j.shape[0] > points_in_i.shape[0] * max_cluster_size_diff_factor:
                if points_in_i.shape[0] > points_in_j.shape[0] * max_cluster_size_diff_factor:
                    points_in_i = get_nearest_points(points_in_i, dip_centers[j], points_in_j.shape[0],
                                                     max_cluster_size_diff_factor, min_sample_size)
                elif points_in_j.shape[0] > points_in_i.shape[0] * max_cluster_size_diff_factor:
                    points_in_j = get_nearest_points(points_in_j, dip_centers[i], points_in_i.shape[0],
                                                     max_cluster_size_diff_factor, min_sample_size)
                points_in_i_or_j = np.append(points_in_i, points_in_j, axis=0)
                proj_points = np.dot(points_in_i_or_j, center_diff)
                _, dip_p_value_2 = dip_test(proj_points)
                dip_p_value = min(dip_p_value, dip_p_value_2)

            # Add pval to dip matrix
            dip_matrix[i][j] = dip_p_value
            dip_matrix[j][i] = dip_p_value

    return dip_matrix


def _adclust_training(X, n_clusters_current, dip_merge_threshold, cluster_loss_weight, ae_weight_loss, centers_cpu,
                      cluster_labels_cpu,
                      dip_matrix_cpu, n_clusters_max, n_clusters_min, dedc_epochs, optimizer, loss_fn, autoencoder,
                      device, trainloader, testloader, debug):
    i = 0
    pred_each_merge = []
    pred_all = []

    while i < dedc_epochs:
        cluster_labels_torch = torch.from_numpy(cluster_labels_cpu).long().to(device)
        centers_torch = torch.from_numpy(centers_cpu).float().to(device)
        dip_matrix_torch = torch.from_numpy(dip_matrix_cpu).float().to(device)

        # Get dip costs matrix
        dip_matrix_eye = dip_matrix_torch + torch.eye(n_clusters_current, device=device)
        dip_matrix_final = dip_matrix_eye / dip_matrix_eye.sum(1).reshape((-1, 1))

        # Iterate over batches
        for batch, ids in trainloader:

            batch_data = batch.to(device)
            embedded = autoencoder.encode(batch_data)
            reconstruction = autoencoder.decode(embedded)
            embedded_centers_torch = autoencoder.encode(centers_torch)

            # Reconstruction Loss
            ae_loss = loss_fn(reconstruction, batch_data)

            # Get distances between points and centers. Get nearest center
            squared_diffs = squared_euclidean_distance(embedded_centers_torch, embedded)

            # Update labels? Pause is needed, so cluster labels can adjust to the new structure

            if i != 0:
                # Update labels
                current_labels = squared_diffs.argmin(1)
                # cluster_labels_torch[ids] = current_labels
            else:
                current_labels = cluster_labels_torch[ids]

            onehot_labels = int_to_one_hot(current_labels, n_clusters_current).float()
            cluster_relationships = torch.matmul(onehot_labels, dip_matrix_final)
            escaped_diffs = cluster_relationships * squared_diffs

            # Normalize loss by cluster distances
            squared_center_diffs = squared_euclidean_distance(embedded_centers_torch, embedded_centers_torch)

            # Ignore zero values (diagonal)
            mask = torch.where(squared_center_diffs != 0)
            masked_center_diffs = squared_center_diffs[mask[0], mask[1]]
            sqrt_masked_center_diffs = masked_center_diffs.sqrt()
            masked_center_diffs_std = sqrt_masked_center_diffs.std() if len(sqrt_masked_center_diffs) > 2 else 0

            # Loss function
            cluster_loss = escaped_diffs.sum(1).mean() * (
                    1 + masked_center_diffs_std) / sqrt_masked_center_diffs.mean()
            cluster_loss *= cluster_loss_weight
            loss = ae_loss * ae_weight_loss + cluster_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update centers
        embedded_data = encode_batchwise(testloader, autoencoder, device)
        embedded_centers_cpu = autoencoder.encode(centers_torch).detach().cpu().numpy()
        cluster_labels_cpu = np.argmin(cdist(embedded_centers_cpu, embedded_data), axis=0)
        optimal_centers = np.array([np.mean(embedded_data[cluster_labels_cpu == cluster_id], axis=0) for cluster_id in
                                    range(n_clusters_current)])
        centers_cpu, embedded_centers_cpu = get_nearest_points_to_optimal_centers(X, optimal_centers, embedded_data)

        # Update Dips
        dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_current)

        if debug:
            print(
                "Iteration {0}  (n_clusters = {4}) - reconstruction loss: {1} / cluster loss: {2} / total loss: {3}".format(
                    i, ae_loss.item(), cluster_loss.item(), loss.item(), n_clusters_current))
            print("max dip", np.max(dip_matrix_cpu), " at ",
                  np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape))

        # i is increased here. Else next iteration will start with i = 1 instead of 0 after a merge
        i += 1

        # Start merging procedure
        dip_argmax = np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape)

        # Is merge possible?
        m = 0
        if i != 0:
            while dip_matrix_cpu[dip_argmax] >= dip_merge_threshold and n_clusters_current > n_clusters_min:
                pred_all.append(np.array(cluster_labels_cpu, dtype='int'))
                if m == 0:
                    pred_each_merge.append(np.array(cluster_labels_cpu, dtype='int'))
                if debug:
                    print("Start merging in iteration {0}.\nMerging clusters {1} with dip value {2}.".format(i,
                                                                                                             dip_argmax,
                                                                                                             dip_matrix_cpu[
                                                                                                                 dip_argmax]))

                # Reset iteration and reduce number of cluster
                i = 0
                n_clusters_current -= 1
                cluster_labels_cpu, centers_cpu, embedded_centers_cpu, dip_matrix_cpu = \
                    _merge_by_dip_value(X, embedded_data, cluster_labels_cpu, dip_argmax, n_clusters_current,
                                        centers_cpu, embedded_centers_cpu)
                dip_argmax = np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape)

                m += 1

        if n_clusters_current == 1:
            if debug:
                print("Only one cluster left")
            break

        if i == dedc_epochs:
            pred_each_merge.append(np.array(cluster_labels_cpu, dtype='int'))
            pred_all.append(np.array(cluster_labels_cpu, dtype='int'))

    return cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder, pred_all, pred_each_merge


def get_trained_autoencoder(trainloader, learning_rate, n_epochs, device, optimizer_class, loss_fn,
                            input_dim, embedding_size, autoencoder_class):
    if embedding_size > input_dim:
        print(
            "WARNING: embedding_size is larger than the dimensionality of the input dataset. Setting embedding_size to",
            input_dim)
        embedding_size = input_dim

    if judge_system():
        act_fn = torch.nn.ReLU
    else:
        act_fn = torch.nn.LeakyReLU

    # Pretrain Autoencoder
    autoencoder = autoencoder_class(input_dim=input_dim, embedding_size=embedding_size,
                                    act_fn=act_fn).to(device)

    optimizer = optimizer_class(autoencoder.parameters(), lr=learning_rate)
    autoencoder.start_training(trainloader, n_epochs, device, optimizer, loss_fn)

    return autoencoder


def _adclust(X, dip_merge_threshold, cluster_loss_weight, ae_weight_loss, n_clusters_max,
             n_clusters_min, batch_size, learning_rate, pretrain_epochs, dedc_epochs, embedding_size,
             debug, optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss()):
    device = detect_device()

    embedded = []
    clusters = []

    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.from_numpy(X).float(), torch.arange(0, X.shape[0]))),
        batch_size=batch_size,
        # sample random mini-batches from the data
        shuffle=True,
        drop_last=False)

    # create a Dataloader to test the autoencoder in mini-batch fashion
    testloader = torch.utils.data.DataLoader(torch.from_numpy(X).float(),
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False)

    autoencoder = get_trained_autoencoder(trainloader, learning_rate, pretrain_epochs, device,
                                          optimizer_class, loss_fn, X.shape[1], embedding_size,
                                          _ADClust_Autoencoder)

    embedded_data = encode_batchwise(testloader, autoencoder, device)

    embedded.append(embedded_data)
    adata_l = sc.AnnData(np.array(embedded_data))
    sc.tl.tsne(adata_l, random_state=0)
    tsne.append(np.array(adata_l.obsm['X_tsne']))

    # Execute Louvain algorithm to get initial micro-clusters in embedded space
    init_centers, cluster_labels_cpu = get_center_labels(embedded_data, resolution=3.0)
    # np.savetxt("./plot/retina/retina_ADClust_pred_init.csv", np.asarray(cluster_labels_cpu), delimiter=',')

    clusters.append(cluster_labels_cpu)

    n_clusters_start = len(np.unique(cluster_labels_cpu))
    print("\n "  "Initialize " + str(n_clusters_start) + "  mirco_clusters \n")

    # Get nearest points to optimal centers
    centers_cpu, embedded_centers_cpu = get_nearest_points_to_optimal_centers(X, init_centers, embedded_data)
    # Initial dip values
    dip_matrix_cpu = _get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_start)

    # Reduce learning_rate from pretraining by a magnitude of 10
    dedc_learning_rate = learning_rate * 0.1
    optimizer = optimizer_class(autoencoder.parameters(), lr=dedc_learning_rate)

    # Start clustering training
    cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder, pred_all, pred_each_merge = _adclust_training(X,
                                                                                                                    n_clusters_start,
                                                                                                                    dip_merge_threshold,
                                                                                                                    cluster_loss_weight,
                                                                                                                    ae_weight_loss,
                                                                                                                    centers_cpu,
                                                                                                                    cluster_labels_cpu,
                                                                                                                    dip_matrix_cpu,
                                                                                                                    n_clusters_max,
                                                                                                                    n_clusters_min,
                                                                                                                    dedc_epochs,
                                                                                                                    optimizer,
                                                                                                                    loss_fn,
                                                                                                                    autoencoder,
                                                                                                                    device,
                                                                                                                    trainloader,
                                                                                                                    testloader,
                                                                                                                    debug)

    embedded_last = encode_batchwise(testloader, autoencoder, device)
    embedded.append(embedded_last)
    clusters.append(cluster_labels_cpu)

    return cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder, \
        pred_all, pred, embedded, tsne, clusters


class ADClust():

    def __init__(self, dip_merge_threshold=0.9, cluster_loss_weight=1, ae_loss_weight=1, batch_size=128,
                 learning_rate=1e-4, pretrain_epochs=100, dedc_epochs=50, embedding_size=10,
                 data_size=10000, n_clusters_max=np.inf, n_clusters_min=3, debug=False):
        self.dip_merge_threshold = dip_merge_threshold
        self.cluster_loss_weight = cluster_loss_weight
        self.ae_loss_weight = ae_loss_weight
        self.n_clusters_max = n_clusters_max
        self.n_clusters_min = n_clusters_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.dedc_epochs = dedc_epochs
        self.embedding_size = embedding_size
        self.debug = debug
        if data_size > 10000:
            self.batch_size = 1024

    def fit(self, X):
        labels, n_clusters, centers, autoencoder, pred_all, pred_each_merge, embedded, tsne, clusters = _adclust(X,
                                                                                                                 self.dip_merge_threshold,
                                                                                                                 self.cluster_loss_weight,
                                                                                                                 self.ae_loss_weight,
                                                                                                                 self.n_clusters_max,
                                                                                                                 self.n_clusters_min,
                                                                                                                 self.batch_size,
                                                                                                                 self.learning_rate,
                                                                                                                 self.pretrain_epochs,
                                                                                                                 self.dedc_epochs,
                                                                                                                 self.embedding_size,
                                                                                                                 self.debug)

        self.labels_ = labels
        self.n_clusters_ = n_clusters
        self.cluster_centers_ = centers
        self.autoencoder = autoencoder

        return labels, n_clusters, pred_all, pred_each_merge, embedded, tsne, clusters
