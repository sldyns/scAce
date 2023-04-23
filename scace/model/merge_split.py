import numpy as np

def cluster_intra_dis(X, cent, label):
    intra_dis = []
    for i in range(len(cent)):
        data_cluster = X[label == i, :]

        cluster_dis = np.linalg.norm(data_cluster - cent[i], axis=0)
        intra_dis.append(np.sum(cluster_dis) / data_cluster.shape[0])

    return intra_dis


def merge_compute(y_pred, mu, scace_emb):

    # Check if len(mu_prepare) == len(np.unique(y_pred))
    idx_cent = []
    for i in range(len(mu)):
        if (y_pred == i).any() == False:
            idx_cent.append(i)
    if len(idx_cent) == 0:
        mu = mu
    else:
        mu = np.delete(mu, idx_cent, 0)
    n_clusters = len(mu)

    # Change label to 0 ~ len(np.unique(label))
    for i in range(len(np.unique(y_pred))):
        if np.unique(y_pred)[i] != i:
            y_pred[y_pred == np.unique(y_pred)[i]] = i


    Centroid = np.array(mu)

    # Compute d_bar
    intra_dis = cluster_intra_dis(scace_emb, Centroid, y_pred)
    d_ave = np.mean(intra_dis)

    sum_1 = 0
    for i in range(0, n_clusters):
        for j in range(i + 1, n_clusters):
            weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2)
            sum_1 += weight * np.linalg.norm(Centroid[i] - Centroid[j])

    d_bar = sum_1 / (n_clusters * (n_clusters - 1))

    return y_pred, Centroid, d_bar, intra_dis, d_ave



def centroid_merge(X, cent_to_merge, label, min_dis, intra_dis, d_ave):

    pred_f = []

    for t in range(200):

        if t == 0:
            pred_f.append(label)

        Cent_dis = []
        Cent_i = []
        Cent_e = []
        Final_Centroid_merge = cent_to_merge

        n_Clusters = len(Final_Centroid_merge)

        for i in range(n_Clusters):
            for e in range(i + 1, n_Clusters):
                weight = d_ave / ((intra_dis[i] + intra_dis[e]) / 2)
                dis = np.linalg.norm(Final_Centroid_merge[i] - Final_Centroid_merge[e])
                Cent_dis.append(weight * dis)
                Cent_i.append(i)
                Cent_e.append(e)

        for i in range(len(Cent_dis)):
            if Cent_dis[i] < min_dis and Cent_dis[i] == min(Cent_dis):

                Cent_merge = (Final_Centroid_merge[Cent_i[i]] + Final_Centroid_merge[Cent_e[i]]) / 2

                Final_Centroid_merge = np.delete(Final_Centroid_merge, (Cent_i[i], Cent_e[i]), 0)
                Final_Centroid_merge = np.insert(Final_Centroid_merge, Cent_i[i], Cent_merge, 0)
                Final_Centroid_merge = np.insert(Final_Centroid_merge, Cent_e[i], 0, 0)

                Final_Centroid_merge = Final_Centroid_merge[~(Final_Centroid_merge == 0).all(axis=1)]

                cent_to_merge = Final_Centroid_merge

                label = np.array(label)
                label[label == Cent_e[i]] = Cent_i[i]

                for i in range(len(np.unique(label))):
                    if np.unique(label)[i] != i:
                        label[label == np.unique(label)[i]] = i
                    else:
                        continue

            else:
                pass

        n_clusters_t = len(np.unique(label))
        pred_f.append(label)

        intra_dis = cluster_intra_dis(X, cent_to_merge, label)
        d_ave = np.mean(intra_dis)

        sum_1 = 0
        for i in range(n_clusters_t):
            for j in range(i + 1, n_clusters_t):
                weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2)

                sum_1 += weight * (np.linalg.norm(Final_Centroid_merge[i] - Final_Centroid_merge[j]))

        d_bar = sum_1 / (n_clusters_t * (n_clusters_t - 1))

        min_dis = d_bar

        count = 0
        for i in range(0, n_clusters_t):
            for j in range(i + 1, n_clusters_t):
                weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2)
                d_inter = weight * np.linalg.norm(Final_Centroid_merge[i] - Final_Centroid_merge[j])

                if d_inter > d_bar:
                    count += 1

        count_true = int((n_clusters_t ** 2 - n_clusters_t) / 2)

        print("-----------------iter: %d-----------------" % int(t + 1))
        print("n_clusters: %d" % n_clusters_t)
        print("count_true: %d" % count_true)
        print("count: %d" % count)

        if count >= count_true:
            print("Reach count!")
            break

        else:
            continue

    return Final_Centroid_merge, label, n_clusters_t, pred_f



def centroid_split(X, X_1, Centroid_split, label):
    """
        Parameters
        ----------
        X
            Embedding after pre-training. Rows are cells and columns are genes.
        X_1
            Embedding + Column vectors of cell types (Label splicing in the last column).
    """

    ### Compute weights
    intra_dis = cluster_intra_dis(X, Centroid_split, label)
    d_ave = np.mean(intra_dis)

    sum_1 = 0
    n_clusters = len(np.unique(label))
    for i in range(0, n_clusters):
        for j in range(i + 1, n_clusters):
            weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2)
            sum_1 += weight * np.linalg.norm(Centroid_split[i] - Centroid_split[j])

    d_bar = sum_1 / (n_clusters * (n_clusters - 1) / 2)
    min_dis = d_bar / 2

    X_copy = 1 * X_1

    Dis_tol = cluster_intra_dis(X, Centroid_split, label)

    for t in range(200):

        idx_split = []
        for i in range(len(Centroid_split)):
            if Dis_tol[i] > min_dis and Dis_tol[i] == max(Dis_tol):
                idx_split.append(i)
                dis = []
                X_2 = np.delete(X_1[X_1[:, -1] == i], -1, 1)
                for m in range(len(X_2)):
                    dis_append = 0
                    for n in range(len(X_2)):
                        dis_append += np.linalg.norm(X_2[m] - X_2[n]) ** 2
                    dis.append(dis_append)
                idx_1 = np.argmin(dis)
                centriod_1 = X_2[idx_1]

                X_3 = np.delete(X_2, idx_1, 0)
                T_m = []
                for m in range(len(X_3)):
                    T_nm = 0
                    for n in range(len(X_3)):
                        D_n = np.linalg.norm(X_3[n] - centriod_1) ** 2
                        d_nm = np.linalg.norm(X_3[m] - X_3[n]) ** 2
                        T_nm += np.maximum(D_n - d_nm, 0)
                    T_m.append(T_nm)
                idx_2 = np.argmax(T_m)
                centriod_2 = X_3[idx_2]

                centroid = np.concatenate(
                    (centriod_1.reshape(len(centriod_1), 1).T, centriod_2.reshape(len(centriod_2), 1).T), axis=0)
                idx_1 = []
                for j in range(len(X_1[X_1[:, -1] == i])):
                    A = np.delete(X_1[X_1[:, -1] == i][j], -1)
                    distance = []
                    for e in range(2):
                        B = centroid[e]
                        D = np.linalg.norm(A - B)
                        distance.append(D)
                    idx = np.argmin(distance)
                    idx_1.append(idx)

                    if idx == 1:
                        if np.unique(label)[0] == 0:
                            idx_a = np.array(np.where(X_1[:, -1] == i))[0, j]
                            a = X_1[X_1[:, -1] == i]
                            a[j, -1] = len(Centroid_split) + i + 1
                            X_copy[idx_a, :] = a[j, :]

                        else:
                            idx_a = np.array(np.where(X_1[:, -1] == i))[0, j]
                            a = X_1[X_1[:, -1] == i]
                            a[j, -1] = len(Centroid_split) + 2 + i
                            X_copy[idx_a, :] = a[j, :]
                    else:
                        if np.unique(label)[0] == 0:
                            idx_a = np.array(np.where(X_1[:, -1] == i))[0, j]
                            a = X_1[X_1[:, -1] == i]
                            a[j, -1] = len(Centroid_split) + i
                            X_copy[idx_a, :] = a[j, :]

                        else:
                            idx_a = np.array(np.where(X_1[:, -1] == i))[0, j]
                            a = X_1[X_1[:, -1] == i]
                            a[j, -1] = len(Centroid_split) + 1 + i
                            X_copy[idx_a, :] = a[j, :]

                Centroid_split = np.concatenate(
                    (Centroid_split, centriod_1.reshape(1, len(centriod_1)), centriod_2.reshape(1, len(centriod_2))))

            else:
                continue

        if len(idx_split) == 0:
            Centroid_split = Centroid_split
            label = label
        else:
            Centroid_split = np.delete(Centroid_split, idx_split, 0)
            label = X_copy[:, -1]
            label = np.array(label)
            for i in range(len(np.unique(label))):
                if np.unique(label)[i] != i:
                    label[label == np.unique(label)[i]] = i
                else:
                    continue
            label = label.tolist()

        n_clusters = Centroid_split.shape[0]
        X_1 = np.concatenate([np.array(X), np.array(label).reshape(len(label), 1)], axis=1)
        X_copy = 1 * X_1

        ### Compute weights
        intra_dis = cluster_intra_dis(X, Centroid_split, np.array(label))
        d_ave = np.mean(intra_dis)

        sum_1 = 0
        for i in range(0, n_clusters):
            for j in range(i + 1, n_clusters):
                weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2)
                sum_1 += weight * np.linalg.norm(Centroid_split[i] - Centroid_split[j])

        d_bar = sum_1 / (n_clusters * (n_clusters - 1) / 2)
        min_dis = d_bar / 2

        Dis_tol = cluster_intra_dis(X, Centroid_split, np.array(label))

        count = 0
        for i in range(len(Dis_tol)):
            if Dis_tol[i] < min_dis:
                count += 1

        print("-----------------iter: %d-----------------" % int(t + 1))
        print("n_clusters: %d" % n_clusters)
        print("count_true: %d" % n_clusters)
        print("count: %d" % count)

        if count >= n_clusters:
            print("Reach count!")
            break

        else:
            continue

    return Centroid_split, label

