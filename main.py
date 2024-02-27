import os
import torch
import  matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import math
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def set_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def loss_curve(train_loss, val_loss, pic):
    if not os.path.exists('figs'):
        os.makedirs('figs')
    fig, ax = plt.subplots()
    ax.plot(range(len(train_loss)),train_loss, label='Train')
    ax.plot(range(len(val_loss)),val_loss, label='Val')
    ax.set_title('Taining Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    text_x = len(train_loss) * 0.5
    text_y = max(max(val_loss), max(train_loss))
    ax.text(
        text_x,
        text_y,
        f"valid loss: {val_loss[-1]:.6f}",
    )
    fig.savefig(f'figs/{pic}.png')
    plt.close(fig)

class MyDataset(Dataset):
    data_record = dict()
    def func(x):
        return torch.sin(x)
    def __init__(self, num_samples, train_ratio, val_ratio, test_ratio, type):
        assert type in ['train', 'val', 'test'], "type must be 'train', 'val' or 'test'"
        train_sample = int(num_samples * train_ratio)
        val_sample = int(num_samples * val_ratio)
        test_sample = num_samples - train_sample - val_sample
        if (train_sample, val_sample, test_sample) in MyDataset.data_record:
            data = MyDataset.data_record[(train_sample, val_sample, test_sample)]
        else:
            if not os.path.exists('data'):
                os.makedirs('data')
            file_name = f'data/{train_sample}_{val_sample}_{test_sample}.pt'
            if os.path.exists(file_name):
                data = torch.load(file_name)
            else:
                data = torch.rand(num_samples, 1, device=device) * 2 * math.pi
                torch.save(data, file_name)
            MyDataset.data_record[(train_sample, val_sample, test_sample)] = data
        # data = torch.rand(num_samples, 1, device=device) * 2 * math.pi
        # data = torch.rand(3000, 1, device=device) * 2 * math.pi
        if type == 'train':
            self.data = data[:train_sample]
        elif type == 'val':
            self.data = data[train_sample:train_sample+val_sample]
        else:
            self.data = data[train_sample+val_sample:]
        self.label = MyDataset.func(self.data)
    def __getitem__(self, index):
        return (self.data[index], self.label[index])
    def __len__(self):
        return len(self.data)

class Net(torch.nn.Module):
    def __init__(self, width, depth, activation, dropout=0.0):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(1, width))
        for _ in range(depth - 2):
            self.layers.append(nn.Sequential(
                nn.Linear(width, width),
                nn.Dropout(dropout),
                nn.BatchNorm1d(width),
                activation
            ))
        if depth >= 2:
            self.layers.append(nn.Linear(width, 1))
        else:
            self.layers.append(nn.Linear(1, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
def train(train_set, val_set, epoches, depth, width,activation,batch_size, pic,lr=1e-3):
    model = Net(width, depth, activation).to(device)
    loss_func = nn.MSELoss()
    train_loss = []
    val_loss = []
    optimaizer = optim.Adam(model.parameters(), lr)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
    train_loss_list = []
    valid_loss_list = []   
    for epoch in range(epoches):
        model.train()
        train_loss = 0
        for k, (data,label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimaizer.zero_grad()
            output = model(data)
            loss = loss_func(output, label)
            loss.backward()
            optimaizer.step()
            train_loss += (loss.item()-train_loss)/(k+1)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for k, (data,label) in enumerate(val_loader):
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = loss_func(output, label)
                val_loss += (loss.item()-val_loss)/(k+1)
            train_loss_list.append(train_loss)
            valid_loss_list.append(val_loss)
            if epoch % 200 == 0:
                print(f"\t\t Epoch: {epoch}, Loss: {loss.item()}")
            if pic is not None:
                loss_curve(train_loss_list, valid_loss_list, pic)
    return model, train_loss, val_loss

def search_hyperparameter(train_set, val_set, num_epoch, batch_size):
    depth_list = [3,5,7]
    width_list = [5,10,15]
    # activation_list = [nn.relu, nn.tanh, nn.sigmoid, nn.leaky_relu, nn.elu]
    activation_list = [nn.ReLU(), nn.ReLU6(), nn.Sigmoid(), nn.Tanh()]
    lr_list = [1e-3, 1e-2,1e-1]
    min_val_loss = 1e10
    parameter_list = []
    for i, depth in enumerate(depth_list):
        for j, width in enumerate(width_list):
            for k, activation in enumerate(activation_list):
                for l, lr in enumerate(lr_list):
                    print(f'Current depth: {depth}, width: {width}, activation: {activation}, lr: {lr}')
                    pic = f'depth_{depth}_width_{width}_activation_{activation}_lr_{lr}.png'
                    model, train_loss, val_loss = train(train_set, val_set, num_epoch, depth, width, activation, batch_size, pic, lr)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_depth = depth
                        best_width = width
                        best_activation = activation
                        best_lr = lr
    return best_depth, best_width, best_activation, best_lr

if __name__ == "__main__":
    seed = 305
    set_seed(seed)
    num_samples = 3000
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    num_epoch = 1000
    batch_size = 1024
    train_set = MyDataset(num_samples, train_ratio, val_ratio, test_ratio, 'train')
    val_set = MyDataset(num_samples, train_ratio, val_ratio, test_ratio, 'val')
    test_set = MyDataset(num_samples, train_ratio, val_ratio, test_ratio, 'test')
    best_depth, best_width, best_activation, best_lr = search_hyperparameter(train_set, val_set, num_epoch, batch_size)
    print(f'Best depth: {best_depth}, width: {best_width}, activation: {best_activation}, lr: {best_lr}')
    pic = f'best_depth_{best_depth}_best_width_{best_width}_best_activation_{best_activation}_best_lr_{best_lr}.png'
    model, train_loss, val_loss = train(train_set, val_set, num_epoch, best_depth, best_width, best_activation, batch_size, pic, best_lr)
    model.eval()
    y_pred = model(test_set.data)
    loss = nn.MSELoss()
    test_loss = loss(y_pred, test_set.label)
    print(f'Test loss: {test_loss.item()}')
    fig, ax = plt.subplots()
    ax.scatter(test_set.data.cpu().numpy(), test_set.label.cpu().numpy(), label='Ground Truth')
    ax.scatter(test_set.data.cpu().numpy(), y_pred.cpu().detach().numpy(), label='Prediction')
    ax.set_title('Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    fig.savefig(f'figs/prediction.png')
    plt.close(fig)




    

        
        