import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.axes import Axes
from models import GNN
from torch_geometric.datasets import PPI


class SamplerNode:
    """sampling the input data, a reimplementation of dropedge"""

    def __init__(self, data_path="../data/ppi"):
        ppi_train = PPI(data_path, split="train")
        self.nfeat = ppi_train.num_node_features
        self.nclass = ppi_train.num_classes

        train_data = ppi_train[0]
        self.train_features = train_data.x
        self.train_labels = train_data.y
        self.train_adj = sp.coo_matrix(
            (np.ones(train_data.num_edges), train_data.edge_index),
            shape=(train_data.num_nodes, train_data.num_nodes),
            dtype=np.float32,
        )

        val_data = PPI(data_path, split="val")[0]
        self.val_features = val_data.x
        self.val_labels = val_data.y
        self.val_adj = sp.coo_matrix(
            (np.ones(val_data.num_edges), val_data.edge_index),
            shape=(val_data.num_nodes, val_data.num_nodes),
            dtype=np.float32,
        )

        test_data = PPI(data_path, split="test")[0]
        self.test_features = test_data.x
        self.test_labels = test_data.y
        self.test_adj = sp.coo_matrix(
            (np.ones(test_data.num_edges), test_data.edge_index),
            shape=(test_data.num_nodes, test_data.num_nodes),
            dtype=np.float32,
        )

    def _process_adj(self, adj, cuda):
        """convert the adj matrix into torch sparse tensor"""
        # convert the sp.coo_matrix into torch.sparse_coo_tensor
        indices = torch.from_numpy(np.vstack((adj.row, adj.col))).long()
        values = torch.from_numpy(adj.data)
        adj = torch.sparse_coo_tensor(indices, values, adj.shape)
        if cuda:
            adj = adj.cuda()
        return adj

    def _process_tensor(self, _tensor, cuda):
        if cuda:
            _tensor = _tensor.cuda()
        return _tensor

    def get_train_labels(self, cuda=True):
        return self._process_tensor(self.train_labels, cuda)

    def get_val_features(self, cuda=True):
        return self._process_tensor(self.val_features, cuda)

    def get_val_labels(self, cuda=True):
        return self._process_tensor(self.val_labels, cuda)

    def get_val_adj(self, cuda=True):
        return self._process_adj(self.val_adj, cuda)

    def get_test_features(self, cuda=True):
        return self._process_tensor(self.test_features, cuda)

    def get_test_labels(self, cuda=True):
        return self._process_tensor(self.test_labels, cuda)

    def get_test_adj(self, cuda=True):
        return self._process_adj(self.test_adj, cuda)

    def randomedge_sampler(self, percent, cuda):
        """
        Randomly drop edge and preserve percent% edges.
        """
        if percent >= 1.0:
            adj = self._process_adj(self.train_adj, cuda)
            features = self._process_tensor(self.train_features, cuda)
            return adj, features

        # get the number of edges
        nnz = self.train_adj.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz * percent)
        perm = perm[:preserve_nnz]
        row = self.train_adj.row[perm]
        col = self.train_adj.col[perm]
        data = self.train_adj.data[perm]
        adj = sp.coo_matrix((data, (row, col)), shape=self.train_adj.shape)
        adj = self._process_adj(adj, cuda)
        features = self._process_tensor(self.train_features, cuda)
        return adj, features


def f1score(output: torch.Tensor, labels: torch.Tensor):
    preds = torch.zeros_like(output)
    preds[output > 0.5] = 1
    f1 = metrics.f1_score(labels, preds, average="macro")
    return f1


def train_once(
    sampler: SamplerNode,
    nlayers: int,
    dropedge: bool,
    pairnorm: bool,
    selfloop: bool,
    activation: str,
):
    model = GNN(
        in_channels=sampler.nfeat,
        out_channels=sampler.nclass,
        hidden_channels=128,
        nb_hidlayers=nlayers,
        activation=activation,
        self_loop=selfloop,
        pair_norm=pairnorm,
    )
    model.init()
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.25, patience=4, min_lr=1e-5, verbose=True
    )

    # record results
    train_loss_list, train_f1_list = [], []
    val_loss_list, val_f1_list = [], []
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        train_adj, train_features = sampler.randomedge_sampler(
            percent=0.9 if dropedge else 1.0, cuda=True
        )
        output = model(train_features, train_adj)
        train_loss = F.binary_cross_entropy_with_logits(
            output, sampler.get_train_labels()
        )
        train_f1 = f1score(output.cpu(), sampler.get_train_labels(False))
        train_loss.backward()
        optimizer.step()
        train_loss_list.append(train_loss.item())
        train_f1_list.append(train_f1.item())

        model.eval()
        with torch.no_grad():
            output = model(sampler.get_val_features(), sampler.get_val_adj())
        val_loss = F.binary_cross_entropy_with_logits(output, sampler.get_val_labels())
        val_f1 = f1score(output.cpu(), sampler.get_val_labels(False))
        val_loss_list.append(val_loss.item())
        val_f1_list.append(val_f1.item())

        if epoch % 10 == 0:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train F1 {:.4f} | Val Loss {:.4f} | Val F1 {:.4f}".format(
                    epoch,
                    train_loss.item(),
                    train_f1.item(),
                    val_loss.item(),
                    val_f1.item(),
                ),
            )
            sys.stdout.flush()
        scheduler.step(val_loss)
    return model, val_f1_list[-1], train_loss_list, val_loss_list


def test(sampler: SamplerNode, model: nn.Module):
    # test
    model.eval()
    with torch.no_grad():
        output = model(sampler.get_test_features(), sampler.get_test_adj())
    test_loss = F.binary_cross_entropy_with_logits(output, sampler.get_test_labels())
    test_f1 = f1score(output.cpu(), sampler.get_test_labels(False))
    print("Test Loss {:.4f} | Test F1 {:.4f}".format(test_loss.item(), test_f1.item()))
    return test_f1


def search_hyperparameter(dataset="ppi"):
    sampler = SamplerNode()
    nlayer_list = [4, 3, 2]
    dropedge_list = [True, False]
    pairnorm_list = [True, False]
    selfloop_list = [True, False]
    activation_list = ["relu", "sigmoid"]

    best_f1 = 0
    for nlayer in nlayer_list:
        for dropedge in dropedge_list:
            for pairnorm in pairnorm_list:
                for selfloop in selfloop_list:
                    for activation in activation_list:
                        params = (nlayer, dropedge, pairnorm, selfloop, activation)
                        print(params)
                        _, val_f1, _, _ = train_once(sampler, *params)
                        if val_f1 > best_f1:
                            best_f1 = val_f1
                            best_params = params
    print("best hyperparamter: ", best_params, f"Val F1: {best_f1:.2f}\n")
    model, _, train_loss, val_loss = train_once(sampler, *best_params)
    test_f1 = test(sampler, model)
    pic_file = os.path.join("..", "figs", "node_classification", f"{dataset}.png")
    os.makedirs(os.path.dirname(pic_file), exist_ok=True)
    fig, ax = plt.subplots()
    ax: Axes
    ax.plot(range(len(train_loss)), train_loss, label="train loss")
    ax.plot(range(len(val_loss)), val_loss, label="valid loss")
    ax.text(0.8 * len(train_loss), 0.8 * max(train_loss), f"test f1: {test_f1:.2f}")
    ax.legend()
    fig.savefig(pic_file)
    return best_params, test_f1


if __name__ == "__main__":
    b = "ppi"
    log_file = os.path.join("..", "log", "node_classification", f"{b}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    f = open(log_file, "w", encoding="utf8")
    sys.stdout = f
    search_hyperparameter(b)
    f.close()
