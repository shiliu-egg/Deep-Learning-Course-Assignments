import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.axes import Axes
from models import GNN
from utils import SamplerNode, accuracy


def train_once(
    sampler: SamplerNode,
    nlayers: int,
    dropedge: bool,
    pairnorm: bool,
    selfloop: bool,
    activation: str,
):
    labels, idx_train, idx_val, _ = sampler.get_label_and_idxes(True)
    adj, features = sampler.randomedge_sampler(percent=1, cuda=True)
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
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        train_adj, train_features = sampler.randomedge_sampler(
            percent=0.9 if dropedge else 1.0, cuda=True
        )
        output = model(train_features, train_adj)
        train_loss_iter = F.cross_entropy(output[idx_train], labels[idx_train])
        train_acc_iter = accuracy(output[idx_train], labels[idx_train])
        train_loss_iter.backward()
        optimizer.step()
        train_loss.append(train_loss_iter.item())
        train_acc.append(train_acc_iter.item())

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
        val_loss_iter = F.cross_entropy(output[idx_val], labels[idx_val])
        val_acc_iter = accuracy(output[idx_val], labels[idx_val])
        val_loss.append(val_loss_iter.item())
        val_acc.append(val_acc_iter.item())

        if epoch % 10 == 0:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}".format(
                    epoch,
                    train_loss_iter.item(),
                    train_acc_iter.item(),
                    val_loss_iter.item(),
                    val_acc_iter.item(),
                ),
            )
            sys.stdout.flush()
        scheduler.step(val_loss_iter)
    return model, val_acc[-1], train_loss, val_loss


def test(sampler: SamplerNode, model: nn.Module):
    # test
    labels, _, _, idx_test = sampler.get_label_and_idxes(True)
    adj, features = sampler.randomedge_sampler(percent=1, cuda=True)
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
    test_loss = F.cross_entropy(output[idx_test], labels[idx_test])
    test_acc = accuracy(output[idx_test], labels[idx_test])
    print(
        "Test Loss {:.4f} | Test Acc {:.4f}".format(test_loss.item(), test_acc.item())
    )
    return test_acc


def search_hyperparameter(dataset="cora"):
    sampler = SamplerNode(dataset=dataset)
    nlayer_list = [4, 3, 2]
    dropedge_list = [True, False]
    pairnorm_list = [True, False]
    selfloop_list = [True, False]
    activation_list = ["relu", "sigmoid"]

    best_acc = 0
    for nlayer in nlayer_list:
        for dropedge in dropedge_list:
            for pairnorm in pairnorm_list:
                for selfloop in selfloop_list:
                    for activation in activation_list:
                        params = (nlayer, dropedge, pairnorm, selfloop, activation)
                        print(params)
                        _, val_acc, _, _ = train_once(sampler, *params)
                        if val_acc > best_acc:
                            best_acc = val_acc
                            best_params = params
    print("best hyperparamter: ", best_params, f"Val Acc: {best_acc:.2f}\n")
    model, _, train_loss, val_loss = train_once(sampler, *best_params)
    test_acc = test(sampler, model)
    pic_file = os.path.join("..", "figs", "node_classification", f"{dataset}.png")
    os.makedirs(os.path.dirname(pic_file), exist_ok=True)
    fig, ax = plt.subplots()
    ax: Axes
    ax.plot(range(len(train_loss)), train_loss, label="train loss")
    ax.plot(range(len(val_loss)), val_loss, label="valid loss")
    ax.text(0.8 * len(train_loss), 0.8 * max(train_loss), f"test acc: {test_acc:.2f}")
    ax.legend()
    fig.savefig(pic_file)
    return best_params, test_acc


if __name__ == "__main__":
    benchmarks = ["cora", "citeseer"]
    for b in benchmarks:
        log_file = os.path.join("..", "log", "node_classification", f"{b}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        f = open(log_file, "w", encoding="utf8")
        sys.stdout = f
        search_hyperparameter(b)
        f.close()
