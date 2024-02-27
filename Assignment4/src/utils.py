import os
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
import sklearn.metrics as metrics
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.utils import negative_sampling


def log(filename, string):
    with open(filename, "a") as f:
        f.write(string + "\n")
    print(string)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def accuracy(output, labels):
    """
    Compute the accuracy of the prediction
    The labels have shape (n,)
    """
    preds = output.max(1)[1].type_as(labels)
    correct = (preds == labels).sum()
    return correct / len(labels)


def roc_auc_compute_fn(y_preds, y_targets):
    """
    Compute the ROC AUC score
    """
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().detach().numpy()
    return metrics.roc_auc_score(y_true, y_pred)


def get_indices(adj, idx_train, idx_val, idx_test, cuda):
    """
    Get the indices of the non-zero elements in adj[idx]
    """
    adj_dense = adj.to_dense()
    train_adj = adj_dense[idx_train][:, idx_train]
    val_adj = adj_dense[idx_val][:, idx_val]
    test_adj = adj_dense[idx_test][:, idx_test]
    train_indices = torch.nonzero(train_adj).t()
    val_indices = torch.nonzero(val_adj).t()
    test_indices = torch.nonzero(test_adj).t()
    if cuda:
        train_indices = train_indices.cuda()
        val_indices = val_indices.cuda()
        test_indices = test_indices.cuda()
    return train_indices, val_indices, test_indices


def data_loader(content_file, cites_file, val_ratio=0.2, test_ratio=0.2, seed=123):
    # Load the content file
    id_cnt, label_cnt = 0, 0
    features, labels = [], []
    id_to_index, label_to_index = {}, {}
    with open(content_file, "r") as f:
        for line in f:
            line = line.strip().split()
            id_to_index[line[0]] = id_cnt
            features.append(line[1:-1])
            if line[-1] not in label_to_index:
                label_to_index[line[-1]] = label_cnt
                label_cnt += 1
            labeli = label_to_index[line[-1]]
            labels.append(labeli)
            id_cnt += 1
    features = torch.from_numpy(np.array(features, dtype=np.float32))
    labels = torch.from_numpy(np.array(labels, dtype=np.int64))

    # Load the cites file
    rows, cols = [], []
    with open(cites_file, "r") as f:
        for line in f:
            line = line.strip().split()
            if line[0] in id_to_index and line[1] in id_to_index:
                p1, p2 = id_to_index[line[0]], id_to_index[line[1]]
                # 构成 p2 -> p1 的边
                rows.append(p2)
                cols.append(p1)

    num_nodes = len(features)
    num_edges = len(rows)
    adj = sp.coo_matrix(
        (np.ones(num_edges), (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )

    # Split the data into train, validation, and test sets
    np.random.seed(seed)
    idx = np.random.permutation(num_nodes)
    num_train = int(num_nodes * (1 - val_ratio - test_ratio))
    num_valid = int(num_nodes * val_ratio)
    idx_train = torch.LongTensor(idx[:num_train])
    idx_val = torch.LongTensor(idx[num_train : num_train + num_valid])
    idx_test = torch.LongTensor(idx[num_train + num_valid :])

    # get the degree of each node from the sp.coo_matrix
    degree = torch.FloatTensor(np.array(adj.sum(axis=1)).squeeze())

    return adj, features, labels, idx_train, idx_val, idx_test, degree


def sp_coo_to_torch_sparse_tensor(sp_coo):
    """convert the sp.coo_matrix into torch.sparse_coo_tensor"""
    indices = torch.from_numpy(np.vstack((sp_coo.row, sp_coo.col))).long()
    values = torch.from_numpy(sp_coo.data)
    shape = torch.Size(sp_coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


class SamplerNode:
    """sampling the input data, a reimplementation of dropedge"""

    def __init__(
        self,
        dataset="cora",
        data_path="../data",
    ):
        dataset = dataset.lower()
        if dataset == "ppi":
            PPI(data_path)
            pass
        else:
            content_pth = os.path.join(data_path, dataset, f"{dataset}.content")
            cites_pth = os.path.join(data_path, dataset, f"{dataset}.cites")
            (
                self.adj,
                self.features,
                self.labels,
                self.idx_train,
                self.idx_val,
                self.idx_test,
                self.degree,
            ) = data_loader(content_pth, cites_pth)

            self.nfeat = self.features.shape[1]
            self.nclass = self.labels.max().item() + 1

    def _process_adj(self, adj, cuda):
        """convert the adj matrix into torch sparse tensor"""
        # convert the sp.coo_matrix into torch.sparse_coo_tensor
        indices = torch.from_numpy(np.vstack((adj.row, adj.col))).long()
        values = torch.from_numpy(adj.data)
        adj = torch.sparse_coo_tensor(indices, values, adj.shape)
        if cuda:
            adj = adj.cuda()
        return adj

    def _process_features(self, features, cuda):
        if cuda:
            features = features.cuda()
        return features

    def randomedge_sampler(self, percent, cuda):
        """
        Randomly drop edge and preserve percent% edges.
        """
        if percent >= 1.0:
            adj = self._process_adj(self.adj, cuda)
            features = self._process_features(self.features, cuda)
            return adj, features

        # get the number of edges
        nnz = self.adj.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz * percent)
        perm = perm[:preserve_nnz]
        row = self.adj.row[perm]
        col = self.adj.col[perm]
        data = self.adj.data[perm]
        adj = sp.coo_matrix((data, (row, col)), shape=self.adj.shape)
        adj = self._process_adj(adj, cuda)
        features = self._process_features(self.features, cuda)
        return adj, features

    def get_label_and_idxes(self, cuda):
        """
        Return all labels and indexes.
        """
        if cuda:
            return (
                self.labels.cuda(),
                self.idx_train.cuda(),
                self.idx_val.cuda(),
                self.idx_test.cuda(),
            )
        else:
            return self.labels, self.idx_train, self.idx_val, self.idx_test


class SamplerEdge:
    """sampling the input data, a reimplementation of dropedge"""

    def __init__(self, dataset: str = "cora", data_path: str = "../data", cuda=True):
        self.device = torch.device("cuda" if cuda else "cpu")
        transform = T.Compose(
            [
                T.NormalizeFeatures(),
                T.ToDevice(self.device),
                T.RandomLinkSplit(
                    num_val=0.05,
                    num_test=0.1,
                    is_undirected=True,
                    add_negative_train_samples=False,
                ),
            ]
        )
        dataset = dataset.lower()
        if dataset == "ppi":
            data_path = os.path.join(data_path, "ppi")
            dataset = PPI(root=data_path, transform=transform)
        else:
            dataset = Planetoid(root=data_path, name=dataset, transform=transform)
        self.train_data, self.val_data, self.test_data = dataset[0]
        self.nfeat = dataset.num_features
        self.num_nodes = self.train_data.num_nodes

    def _process_adj(self, edge_index, cuda):
        """convert the adj matrix into torch sparse tensor"""
        # convert the sp.coo_matrix into torch.sparse_coo_tensor
        indices = edge_index
        values = torch.ones(indices.shape[1])
        if cuda:
            indices = indices.cuda()
            values = values.cuda()
        adj = torch.sparse_coo_tensor(
            indices, values, size=(self.num_nodes, self.num_nodes), dtype=torch.float32
        )
        if cuda:
            adj = adj.cuda()
        return adj

    def _process_features(self, features, cuda):
        if cuda:
            features = features.cuda()
        return features

    def randomedge_sampler(self, percent, cuda):
        """
        Randomly drop edge and preserve percent% edges.
        """
        if percent >= 1.0:
            adj = self._process_adj(self.train_data.edge_index, cuda)
            features = self._process_features(self.train_data.x, cuda)
            return adj, features

        # get the number of edges
        nnz = self.train_data.edge_index.shape[1]
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz * percent)
        perm = perm[:preserve_nnz]
        edge_index = self.train_data.edge_index[:, perm]
        adj = self._process_adj(edge_index, cuda)
        features = self._process_features(self.train_data.x, cuda)
        return adj, features

    def get_feature_and_adj(self, data, cuda):
        """
        Return the feature and adj matrix.
        """
        adj = self._process_adj(data.edge_index, cuda)
        features = self._process_features(data.x, cuda)
        return adj, features

    def get_edge_index_label(self, data, cuda):
        """
        Return the edge index and labels.
        """
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.shape[1],
        )
        edge_label_index = data.edge_label_index
        if cuda:
            neg_edge_index = neg_edge_index.cuda()
            edge_label_index = edge_label_index.cuda()
        edge_label_index = torch.cat([edge_label_index, neg_edge_index.cuda()], dim=1)
        neg_label = torch.zeros(neg_edge_index.shape[1])
        edge_label = data.edge_label
        if cuda:
            edge_label = data.edge_label.cuda()
            neg_label = neg_label.cuda()
        edge_label = torch.cat([edge_label, neg_label], dim=0)
        if cuda:
            edge_label_index = edge_label_index.cuda()
            edge_label = edge_label.cuda()
        return edge_label_index, edge_label
