import os
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def load_net_Z(args):
    net_path = os.path.join(args.data_path, args.data_log)
    z_path = os.path.join(args.logs_path, args.z_log, "Z.npy")
    net = np.load(net_path + "_net.npy")
    occ = np.load(net_path + "_occ.npy")
    z = np.load(z_path)
    # 将有向图转化为无向图
    index = np.where(net == 1)
    undirected_net = np.zeros_like(net)
    for (i, j) in zip(index[0], index[1]):
        undirected_net[i][j] = undirected_net[j][i] = 1
    return net, z, occ, undirected_net


def find_possible_node_pair(occ, net, occ_threshold, filter, analysis):
    """筛选出net中有边的共现率的分布情况"""
    index = np.where(net == 1)
    occ_rate = occ[index]
    if analysis:
        occ_rate1 = sorted(occ_rate)
        print(occ_rate1[:10])
        plot_distribution(occ_rate)
    if filter:
        in_filter = np.where(occ > occ_threshold, 1, 0)
        possible_node_pair_index = np.where(occ > occ_threshold)
        return possible_node_pair_index
    else:
        return None  # 不进行过滤


def plot_distribution(dis):
    """画一下距离的直方图"""
    plt.hist(dis.flatten())
    plt.show()


def ED(num_nodes, z, node_pair_index):
    """欧式距离计算"""
    dis = np.ones((num_nodes, num_nodes))
    if node_pair_index == None:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dis[i][j] = dis[j][i] = np.linalg.norm(z[i] - z[j])  # 因为对称性，计算一半即可
    else:
        for (i, j) in zip(node_pair_index[0], node_pair_index[1]):
            dis[i][j] = np.linalg.norm(z[i] - z[j])
    return dis


def MD(num_nodes, z, node_pair_index):
    """曼哈顿距离"""
    dis = np.ones((num_nodes, num_nodes))
    if node_pair_index == None:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dis[i][j] = dis[j][i] = np.linalg.norm(z[i] - z[j], ord=1)  # 因为对称性，计算一半即可
    else:
        for (i, j) in zip(node_pair_index[0], node_pair_index[1]):
            dis[i][j] = np.linalg.norm(z[i] - z[j], ord=1)  # 因为对称性，计算一半即可
    return dis


def cosine_dis(num_nodes, z, node_pair_index):
    """余弦相似度"""
    # 在[-1,1]之间，值越趋近于1，代表两个向量的方向越接近；越趋近于-1，他们的方向越相反；接近于0，表示两个向量近乎于正交。
    dis = np.zeros((num_nodes, num_nodes))
    if node_pair_index == None:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # 因为对称性，计算一半即可
                dis[i][j] = dis[j][i] = np.dot(z[i], z[j]) / (np.linalg.norm(z[i]) * (np.linalg.norm(z[j])))
    else:
        for (i, j) in zip(node_pair_index[0], node_pair_index[1]):
            dis[i][j] = np.dot(z[i], z[j]) / (np.linalg.norm(z[i]) * (np.linalg.norm(z[j])))
    return dis


def KL(num_nodes, z, node_pair_index):
    """KL散度，不满足对称性"""
    # KL散度的取值范围是[0,+∞] P和Q越相似，KL散度越小
    dis = np.ones((num_nodes, num_nodes))
    if node_pair_index == None:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dis[i][j] = scipy.stats.entropy(z[i], z[j])
    else:
        for (i, j) in zip(node_pair_index[0], node_pair_index[1]):
            dis[i][j] = scipy.stats.entropy(z[i], z[j])
    return dis


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def JS(num_nodes, z, node_pair_index):
    """JS散度"""
    # JS散度的取值在0-1之间，完全相同为0
    dis = np.ones((num_nodes, num_nodes))
    if node_pair_index == None:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dis[i][j] = dis[j][i] = JS_divergence(z[i], z[j])
    else:
        for (i, j) in zip(node_pair_index[0], node_pair_index[1]):
            dis[i][j] = JS_divergence(z[i], z[j])
    return dis


def WS(num_nodes, z, node_pair_index):
    """wasserstein_distance"""
    # 如果两个分布重合，则为0
    dis = np.ones((num_nodes, num_nodes))
    if node_pair_index == None:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dis[i][j] = dis[j][i] = scipy.stats.wasserstein_distance(z[i], z[j])
    else:
        for (i, j) in zip(node_pair_index[0], node_pair_index[1]):
            dis[i][j] = scipy.stats.wasserstein_distance(z[i], z[j])
    return dis


def prediction(dis, threshold):
    pred_net = np.where(dis < threshold, 1, 0)
    return pred_net


def prediction_cos(dis, threshold):
    pred_net = np.where(dis > threshold, 1, 0)
    return pred_net


def metrics(pred, true, is_print):
    num_node = pred.shape[0]
    # Accuracy
    accuracy = 1 - (np.sum(np.abs(pred - true)) / (np.sum(pred) + np.sum(true)))
    # F1
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(num_node):
        for j in range(num_node):
            if true[i][j] == 1 and pred[i][j] == 1:
                tp += 1
            elif true[i][j] == 0 and pred[i][j] == 1:
                fp += 1
            elif true[i][j] == 1 and pred[i][j] == 0:
                fn += 1
            elif true[i][j] == 0 and pred[i][j] == 0:
                tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    if is_print:
        print("accuracy: {:.5f}".format(accuracy),
              "precision: {:.5f}".format(precision),
              "recall: {:.5f}".format(recall),
              "f1: {:.5f}".format(f1))
        print("混淆矩阵\n", np.array([[tn, fn], [fp, tp]]))
    return accuracy, precision, recall, f1


def plot_metrics(dis, thre_down, thre_up, step, net, distance):
    x, acc, pre, rec, f1 = [], [], [], [], []
    threshold = thre_down
    while threshold <= thre_up:
        threshold += step
        x.append(threshold)
        if distance == 'cos':
            pred = prediction_cos(dis, threshold)
        else:
            # 指标越小越好
            pred = prediction(dis, threshold)
        accuracy, precision, recall, f1_score = metrics(pred, net, is_print=False)
        acc.append(accuracy)
        pre.append(precision)
        rec.append(recall)
        f1.append(f1_score)

    plt.plot(x, acc, linewidth=2.0, linestyle='--', marker="o", label="Accuracy")
    plt.plot(x, pre, linewidth=2.0, linestyle='--', marker="D", label="Precision")
    plt.plot(x, rec, linewidth=2.0, linestyle='--', marker="*", label="Recall")
    plt.plot(x, f1, linewidth=2.0, linestyle='--', marker="x", label="F1_score")
    plt.legend()
    plt.title(distance)
    plt.xlabel("Threshold")
    plt.ylabel("Metrics")
    plt.show()
