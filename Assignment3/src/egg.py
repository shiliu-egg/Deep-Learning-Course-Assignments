import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import time
from torch.optim.lr_scheduler import LambdaLR
import os
import matplotlib.pyplot as plt


def set_seed(seed: int = 0):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def calTop1Acc(model, dataLoader):
    model.eval()
    acc = 0
    with torch.no_grad():
        for x, y in dataLoader:
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            output = model(x)
            acc += (output.argmax(dim=1) == y).sum().item()
    return acc / len(dataLoader.dataset)


def draw(trainLossList, validLossList):
    fig, ax = plt.subplots()
    ax.set_title("Training and Validation Loss")
    ax.plot(range(len(trainLossList)), trainLossList, label="Train Loss")
    ax.plot(range(len(validLossList)), validLossList, label="Validation Loss")
    ax.legend()
    figDir = os.path.join("..", "figs")
    if not os.path.exists(figDir):
        os.makedirs(figDir, exist_ok=True)
    fig.savefig(os.path.join(figDir, "trainLoss.png"), dpi=100)
    plt.close(fig)

class RNN(nn.Module):
    def __init__(self, vocab_size, n_layers, dropout, norm, residual, embedding_dim=1024, hidden_dim=1024, output_dim=5):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.residual = residual
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Sequential(
                nn.LSTM(embedding_dim, hidden_dim, batch_first=True),
                nn.LayerNorm(hidden_dim) if norm else nn.Identity(),
                nn.Dropout(0.5) if dropout else nn.Identity(),
                nn.ReLU()
            )
            self.layers.append(layer)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_res = embedded
        output = embedded
        for layer in self.layers:
            output, _ = layer[0](output)  
            output = layer[1:](output)  
            if self.residual:
                output = output + embedded_res  
        output = self.fc(output[:, -1, :])
        return output


class ReviewsDataset(Dataset):
    def __init__(self, texts, stars, seq_length=100):
        self.texts = texts
        self.stars = stars
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if len(text) > self.seq_length:
            text = text[:self.seq_length]
        else:
            text += [vocab_to_int['<PAD>']] * (self.seq_length - len(text))
        return torch.tensor(text, dtype=torch.long), torch.tensor(self.stars[idx], dtype=torch.long)



def trainOnce(vocab_size, n_layers, dropout, norm, residual, lr_decay):
    model = RNN(vocab_size, n_layers, dropout, norm, residual).to(device)
    train_texts, val_texts, train_stars, val_stars = train_test_split(texts_ints, stars, test_size=0.2, random_state=42)

    train_dataset = ReviewsDataset(train_texts, train_stars)
    val_dataset = ReviewsDataset(val_texts, val_stars)

    batch_size = 2056
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if lr_decay:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)

    train_loss_list = []
    valid_loss_list = []
    for epoch in range(10):
        start = time.time()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_dataset)
        train_loss_list.append(train_loss)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device, dtype=torch.long)
                output = model(x)
                loss = criterion(output, y)
                valid_loss += loss.item() * len(x)
            valid_loss /= len(val_dataset)
            valid_loss_list.append(valid_loss)
        end = time.time()
        print(
            f"epoch {epoch}, train loss {train_loss}, valid loss {valid_loss}, time {end-start}"
        )
        if lr_decay:
            scheduler.step()

    acc = calTop1Acc(model, val_loader)
    return train_loss_list, valid_loss_list, acc, model

def searchHyperParameter(vocab_size):
    LayerList = [6,5,4]
    dropoutList = [True, False]
    NormList = [True, False]
    residualList = [True, False]
    lrDecayList = [True, False]
    # LayerList = [4]
    # dropoutList = [False]
    # NormList = [False]
    # residualList = [False]
    # lrDecayList = [False]
    bestAcc = 0
    bestParam = None
    for n_layer in LayerList:
        for dropout in dropoutList:
            for Norm in NormList:
                for residual in residualList:
                    for lrDecay in lrDecayList:
                        param = (vocab_size, n_layer, dropout, Norm, residual, lrDecay)
                        print(param)
                        _, _, acc, _ = trainOnce(*param)
                        print(f"ACC {acc}\n")
                        if acc > bestAcc:
                            bestAcc = acc
                            bestParam = param
    print("\nbest Param:", bestParam)
    print("best Acc:", bestAcc)
    return bestParam


if __name__ == "__main__":
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    set_seed(3407)


    with open('../dataset/yelp_academic_dataset_review.json', 'r') as f:
        data = [json.loads(line) for line in f]
    texts = [d['text'] for d in data]
    stars = [d['stars'] - 1 for d in data] 

    words = ' '.join(texts).split()
    word_counts = Counter(words)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: idx + 2 for idx, word in enumerate(sorted_words)}  
    vocab_to_int['<PAD>'] = 0  
    vocab_to_int['<UNK>'] = 1  


    texts_ints = [[vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()] for text in texts]
    vocab_size = len(vocab_to_int)



    bestParam = searchHyperParameter(vocab_size)


    train_loss_list, valid_loss_list, acc, model = trainOnce(*bestParam)


    draw(train_loss_list, valid_loss_list)


    with open('../dataset/test.json', 'r') as f:
        test_data = [json.loads(line) for line in f]
    test_texts = [d['text'] for d in test_data]
    test_stars = [d['stars']-1 for d in test_data]
    test_texts_ints = [[vocab_to_int.get(word, vocab_size) for word in text.split()] for text in test_texts]
    test_dataset = ReviewsDataset(test_texts_ints, test_stars)
    test_loader = DataLoader(test_dataset, batch_size=2056, shuffle=False)
    test_acc = calTop1Acc(model, test_loader)
    print("Test Accuracy:", test_acc)
