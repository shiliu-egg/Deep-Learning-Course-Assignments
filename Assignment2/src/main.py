import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader
from typing import List
from model import MyModule
# from model_zj import CNN_net
from data import TinyImagenet
import os
import matplotlib.pyplot as plt
import time
from torchvision.models import resnet34, resnet50, resnet18, resnet101,resnet152

device = "cuda:5"


def set_seed(seed: int = 0):
    import sys

    if "random" in sys.modules:
        import random

        random.seed(seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    if "numpy" in sys.modules:
        import numpy

        numpy.random.seed(seed)


def calTop1Acc(model: nn.Module, dataLoader: DataLoader):
    with torch.no_grad():
        model.eval()
        acc = 0
        for x, y in dataLoader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            acc += (output.argmax(dim=1) == y).sum().item()
        acc /= len(dataLoader.dataset)
    return acc


"""
1. dropout
2. normalization
3. learning rate decay
4. residual connection
5. network depth
"""


def trainOnce(
    depth: int, dropout: bool, normalize: bool, has_residual: bool, lrDecay: float
):
    epochs = 10
    # model = MyModule(depth, dropout, normalize, has_residual)
    model = resnet50()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    if lrDecay:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)
    trainData = TinyImagenet(type="train")
    validData = TinyImagenet(type="valid")

    batchSize = 512
    trainLoader = DataLoader(
        trainData,
        batch_size=batchSize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    validLoader = DataLoader(
        validData,
        batch_size=batchSize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    trainLossList = []
    validLossList = []
    model.to(device)
    for epoch in range(epochs):
        start = time.time()
        model.train()
        trainAvgLoss = 0
        for pic, label in trainLoader:
            pic = pic.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(pic)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            trainAvgLoss += loss.item() * batchSize
        trainAvgLoss /= len(trainData)
        trainLossList.append(trainAvgLoss)

        with torch.no_grad():
            model.eval()
            validAvgLoss = 0
            for pic, label in validLoader:
                pic = pic.to(device)
                label = label.to(device)
                output = model(pic)
                loss = criterion(output, label)
                validAvgLoss += loss.item() * batchSize
            validAvgLoss /= len(validData)
            validLossList.append(validAvgLoss)
        end = time.time()
        print(
            f"epoch {epoch}, train loss {trainAvgLoss}, valid loss {validAvgLoss}, time {end-start}"
        )

        if lrDecay:
            scheduler.step()
    acc = calTop1Acc(model, trainLoader)
    return trainLossList, validLossList, acc, model


def searchHyperParameter():
    depthList = [6, 5, 4]
    normalizeList = [True, False]
    dropoutList = [True, False]
    residualList = [True, False]
    lrDecayList = [True, False]
    bestAcc = 0
    bestParam = None
    for depth in depthList:
        for dropout in dropoutList:
            for normalize in normalizeList:
                for residual in residualList:
                    for lrDecay in lrDecayList:
                        param = (depth, dropout, normalize, residual, lrDecay)
                        print("(depth, dropout, normalize, residual, lrDecay)")
                        print(param)
                        _, _, acc, _ = trainOnce(*param)
                        print(f"ACC {acc}\n")
                        if acc > bestAcc:
                            bestAcc = acc
                            bestParam = param
    print("\nbest Param:", bestParam)
    print("best Acc:", bestAcc)
    return bestParam


def test(model: nn.Module):
    model = model.to(device)
    testData = TinyImagenet(type="test")
    batchSize = 64
    testLoader = DataLoader(
        testData,
        batch_size=batchSize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    acc = calTop1Acc(model, testLoader)
    return acc


def draw(trainLossList: List[float], validLossList: List[float]):
    fig, ax = plt.subplots()
    ax.set_title("Training Loss")
    ax.plot(range(len(trainLossList)), trainLossList, label="train")
    ax.plot(range(len(validLossList)), validLossList, label="valid")
    ax.legend()
    text_x = len(trainLossList) * 0.5
    text_y = max(trainLossList)
    ax.text(
        text_x,
        text_y,
        f"valid loss: {trainLossList[-1]:.6f}",
    )
    figDir = os.path.join("..", "figs")
    if not os.path.exists(figDir):
        os.makedirs(figDir)
    fig.savefig(os.path.join(figDir, "trainLoss.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    set_seed(3407)
    torch.autograd.set_detect_anomaly(True)
    bestParam = searchHyperParameter()
    # bestParam = (4, True, True, True, True)
    trainLossList, validLossList, acc, model = trainOnce(*bestParam)
    print("Train ACC:", acc)
    acc = test(model)
    
    draw(trainLossList, validLossList)
