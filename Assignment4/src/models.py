import torch
import torch.nn.functional as F
import torch.nn as nn

class Dense(nn.Module):
    def __init__(self, in_channels, out_channels, activation = "relu"):
        super(Dense, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.activation = activation
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        elif self.activation == "sigmoid":
            x = F.sigmoid(x)
        else:
            raise ValueError("Activation function not supported")
        x = self.linear2(x)
        return x
    
class GraphConv(nn.Module):
    """
    GCN Layer
    """

    def __init__(self, in_features, out_features, activation="relu",
                  self_loop=True, pair_norm=False):
        """
        :param in_features: input feature dimension
        :param out_features: output feature dimension
        :param activation: activation function type
        :param self_loop: whether to add self feature modeling.
        :param pair_norm: whether to apply pair normalization
        """
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.self_loop = self_loop
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        else:
            raise ValueError("Activation function not supported")
        
        if self.self_loop:
            self.self_weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
            nn.init.xavier_uniform_(self.self_weight)
        else:
            self.self_weight = None

        if pair_norm:
            self.pn = PairNorm()
        else:
            self.pn = None
        
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        if self.self_loop:
            num_nodes = x.size(0)
            self_loop = torch.eye(num_nodes)
            if x.is_cuda:
                self_loop = self_loop.cuda()
            adj = self_loop + adj
        output = torch.spmm(adj, support)
        if self.self_weight is not None:
            output = output + torch.mm(x, self.self_weight)
        output = self.activation(output)
        if self.pn is not None:
            output = self.pn(output)
        return output
    
class PairNorm(nn.Module):
    """
    PairNorm Layer (PN)
    """

    def __init__(self, scale=1.0):
        """
        :param scale: initial scaling factor
        """
        super(PairNorm, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = x - torch.mean(x, dim=0, keepdim=True)
        rownorm_mean = torch.sqrt(torch.mean(torch.pow(x, 2).sum(dim=1) + 1e-6))
        x = self.scale * x / rownorm_mean
        return x
    
class input_layer(nn.Module):
    def __init__(self, in_channels, out_channels, activation = "relu"):
        super(input_layer, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.activation = activation
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        elif self.activation == "sigmoid":
            x = F.sigmoid(x)
        else:
            raise ValueError("Activation function not supported")
        x = self.linear2(x)
        return x
    
class output_layer(nn.Module):
    def __init__(self, in_channels, out_channels, activation = "relu"):
        super(output_layer, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.activation = activation
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        elif self.activation == "sigmoid":
            x = F.sigmoid(x)
        else:
            raise ValueError("Activation function not supported")
        x = self.linear2(x)
        return x
    

class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, nb_hidlayers,
                  activation="relu", self_loop=True, pair_norm=False):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.nb_hidlayers = nb_hidlayers
        self.activation = activation
        self.self_loop = self_loop
        self.pair_norm = pair_norm

        self.input_layer = input_layer(self.in_channels, self.hidden_channels, self.activation)
        self.output_layer = output_layer(self.hidden_channels, self.out_channels, self.activation)
        self.hid_layers = nn.ModuleList()
        for i in range(self.nb_hidlayers):
            self.hid_layers.append(GraphConv(self.hidden_channels, self.hidden_channels, 
                                             self.activation, self.self_loop, self.pair_norm))

    def forward(self, x, adj):
        x = self.input_layer(x)
        for hid_layer in self.hid_layers:
            x = hid_layer(x, adj)
        x = self.output_layer(x)
        return x

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

class link_prediction(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, nb_hidlayers,
                  activation="relu", self_loop=True, pair_norm=False):
        super(link_prediction, self).__init__()
        self.gnn = GNN(in_channels, out_channels, hidden_channels, nb_hidlayers,
                    activation, self_loop, pair_norm)
        self.gnn.init()
        
    def encode(self, x, adj):
        return self.gnn(x, adj)
    
    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)