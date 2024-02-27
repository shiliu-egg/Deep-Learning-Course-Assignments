import os
import sys

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.axes import Axes
from models import link_prediction
from utils import SamplerEdge


def train_once(
    sampler: SamplerEdge,
    nlayers: int,
    dropedge: bool,
    pairnorm: bool,
    selfloop: bool,
    activation: str,
):
    val_adj, val_features = sampler.get_feature_and_adj(sampler.val_data, True)
    val_index = sampler.val_data.edge_label_index
    val_label = sampler.val_data.edge_label

    model = link_prediction(
        in_channels=sampler.nfeat,
        out_channels=64,
        hidden_channels=128,
        nb_hidlayers=nlayers,
        activation=activation,
        self_loop=selfloop,
        pair_norm=pairnorm,
    )
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.25, patience=4, min_lr=1e-5, verbose=True
    )

    # record results
    train_loss_list = []
    val_loss_list = []
    val_auc_list = []

    # train
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        train_adj, train_features = sampler.randomedge_sampler(
            percent=0.9 if dropedge else 1.0, cuda=True
        )
        train_index, train_label = sampler.get_edge_index_label(
            sampler.train_data, True
        )
        embedding = model.encode(train_features, train_adj)
        output = model.decode(embedding, train_index).view(-1)
        train_loss = F.binary_cross_entropy_with_logits(output, train_label)
        train_loss.backward()
        optimizer.step()
        train_loss_list.append(train_loss.item())
        pred = (output > 0).int()
        train_acc = (pred == train_label).float().mean()  #! float 不可以进行等号判断

        model.eval()
        with torch.no_grad():
            val_embedding = model.encode(val_features, val_adj)
            val_output = model.decode(val_embedding, val_index).view(-1)
        val_loss = F.binary_cross_entropy_with_logits(val_output, val_label)
        val_loss_list.append(val_loss.item())
        val_auc = metrics.roc_auc_score(val_label.cpu(), val_output.cpu())
        val_auc_list.append(val_auc)

        if epoch % 10 == 0:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train ACC {:.4f} | Val Loss {:.4f} | Val AUC {:.4f}".format(
                    epoch, train_loss, train_acc, val_loss, val_auc
                )
            )
            sys.stdout.flush()
        scheduler.step(val_loss)
    return model, val_auc_list[-1], train_loss_list, val_loss_list


def test(sampler: SamplerEdge, model: nn.Module):
    # test
    # labels, _, _, idx_test = sampler.get_label_and_idxes(True)
    # adj, features = sampler.randomedge_sampler(percent=1, cuda=True)
    test_adj, test_features = sampler.get_feature_and_adj(sampler.test_data, True)
    test_index = sampler.test_data.edge_label_index
    test_label = sampler.test_data.edge_label
    model.eval()
    with torch.no_grad():
        embedding_test = model.encode(test_features, test_adj)
        output = model.decode(embedding_test, test_index).view(-1)
    test_loss = F.binary_cross_entropy_with_logits(output, test_label)
    test_auc = metrics.roc_auc_score(test_label.cpu(), output.cpu())
    print(
        "Test Loss {:.4f} | Test Acc {:.4f}".format(test_loss.item(), test_auc.item())
    )
    return test_auc


def search_hyperparameter(dataset="cora"):
    sampler = SamplerEdge(dataset=dataset)
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
    print("best hyperparamter: ", best_params, f"Val Auc: {best_acc:.2f}\n")
    model, _, train_loss, val_loss = train_once(sampler, *best_params)
    test_auc = test(sampler, model)
    pic_file = os.path.join("..", "figs", "link_prediction", f"{dataset}.png")
    os.makedirs(os.path.dirname(pic_file), exist_ok=True)
    fig, ax = plt.subplots()
    ax: Axes
    ax.plot(range(len(train_loss)), train_loss, label="train loss")
    ax.plot(range(len(val_loss)), val_loss, label="train loss")
    ax.text(0.8 * len(train_loss), 0.8 * max(train_loss), f"test auc: {test_auc:.2f}")
    ax.legend()
    fig.savefig(pic_file)
    return best_params, test_auc


if __name__ == "__main__":
    benchmarks = [
        # "cora",
        "citeseer",
    ]
    for b in benchmarks:
        log_file = os.path.join("..", "log", "link_prediction", f"{b}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        f = open(log_file, "w", encoding="utf8")
        sys.stdout = f
        search_hyperparameter(b)
        f.close()
