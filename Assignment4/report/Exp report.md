# 实验要求
 使用 pytorch 或者 tensorflow 的相关神经网络库，编写图卷积神经网络模型 (GCN)，并在相应的图结构 数据集上完成节点分类和链路预测任务，最后分析自环、层数、DropEdge 、PairNorm 、激活函数等 因素对模型的分类和预测性能的影响 。  
# 实验过程
## 数据处理
本次实验使用的数据集为Cora 、Citeseer 、PPI数据集，在数据处理中，首先将数据集划分
为训练集，验证集和测试集，结点分类任务中，按节点将数据集进行划分，其中训练集，验证集，测试集的比
例为6：2：2。
首先，我们给出节点分类的数据加载代码，如下所示：
```python
def data_loader(content_file, cites_file, val_ratio=0.2, test_ratio=0.2, seed=123):
    # Load the content file
    with open(content_file, 'r') as f:
        content = f.readlines()
    content = [x.strip().split() for x in content]
    content = np.array(content)

    # Create a dictionary that maps paper IDs to node indices
    id_to_index = {paper_id: i for i, paper_id in enumerate(content[:, 0])}

    features = content[:, 1:-1].astype(np.float32)
    labels = content[:, -1]
    label_set = set(labels)
    # encode the lables into one-hot vectors
    label_to_index = {label: i for i, label in enumerate(label_set)}
    labels = np.array([label_to_index[label] for label in labels], dtype=np.int64)

    # Load the cites file
    with open(cites_file, 'r') as f:
        cites = f.readlines()
    cites = [x.strip().split() for x in cites]
    cites = np.array([[id_to_index[x[0]], id_to_index[x[1]]] for x in cites], 
                     dtype=np.int64)
    num_nodes = features.shape[0]
    adj = sp.coo_matrix((np.ones(cites.shape[0]), (cites[:, 1], cites[:, 0])), 
                        shape=(num_nodes, num_nodes), dtype=np.float32)

    # Split the data into train, validation, and test sets
    np.random.seed(seed)
    idx = np.random.permutation(num_nodes)
    idx_train = idx[:int(num_nodes*(1-val_ratio-test_ratio))]
    idx_val = idx[int(num_nodes*(1-val_ratio-test_ratio)):int(num_nodes*(1-test_ratio))]
    idx_test = idx[int(num_nodes*(1-test_ratio)):]

    # Extract the required data format 
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # get the degree of each node from the sp.coo_matrix
    degree = torch.FloatTensor(np.array(adj.sum(1)))

    return adj, features, labels, idx_train, idx_val, idx_test, degree

```
链路预测需要根据边对数据集进行划分，其中训练集、验证集和测试集的比例为8.5:0.5:1。我们使用 torch_geometric 中的 RandomLinkSplit 函数来执行数据集的划分。数据集划分的代码封装在 Sampler_edge 类中，以下是关于数据集划分的具体代码：
```python
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 transform = T.Compose([
 T.NormalizeFeatures(),
 T.ToDevice(device),
 T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
 add_negative_train_samples=False),
 ])
 path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
'Planetoid')
 dataset = Planetoid(path, name='Cora', transform=transform)
 # After applying the `RandomLinkSplit` transform, the data is transformed from
 # a data object to a list of tuples (train_data, val_data, test_data), with
 # each element representing the corresponding split.
 train_data, val_data, test_data = dataset[0]

```
## 模型结构
首先构建GCN卷积模型
```python
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
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, 
                                                               out_features))
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
            self.self_weight = nn.parameter.Parameter(torch.FloatTensor(in_features, 
                                                                        out_features))
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
```
 其中，pair_norm为对图卷积层的输出进行归一化，代码如下：  
```python
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
```
DropEdge的实现已被整合进Sampler类中，与DropEdge相关的代码段在Sampler类中如下所示：
```python
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
 preserve_nnz = int(nnz*percent)
 perm = perm[:preserve_nnz]
 edge_index = self.train_data.edge_index[:, perm]
 adj = self._process_adj(edge_index, cuda)
 features = self._process_features(self.train_data.x, cuda)
 return adj, features

```
除了图卷积层外，我们再为输入和输出分别构造一个FFN，代码如下：
```python
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
```
现在，我们呈现 GNN 的整体结构，代码如下所示：
```python
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

        self.input_layer = input_layer(self.in_channels, self.hidden_channels, 
                                       self.activation)
        self.output_layer = output_layer(self.hidden_channels, self.out_channels, 
                                         self.activation)
        self.hid_layers = nn.ModuleList()
        for i in range(self.nb_hidlayers):
            self.hid_layers.append(GraphConv(self.hidden_channels, 
                                             self.hidden_channels, 
                                             self.activation, s
                                             elf.self_loop, 
                                             self.pair_norm))

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
```
针对链路预测任务，我们额外定义了一个名为link_prediction的类。该类包含两个函数：encoder和decoder，分别用于对输入图进行编码和解码。以下是代码示例：
```python
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

```
模型的详细训练过程请参考代码文件 node_classification.py 和 link_prediction.py。
## 超参数搜索及实验结果
我们编写了一个sh文件用以超参数搜索
```python
#!/bin/bash

layers=(1 2 3 4)
pair_norm=(True False)
self_loop=(True False)
activations=("relu" "tanh" "sigmoid")

for layer in "${layers[@]}"; do
    for pn in "${pair_norm[@]}"; do
        for sl in "${self_loop[@]}"; do
            for act in "${activations[@]}"; do
                echo "Running with layer=$layer, pair_norm=$pn, self_loop=$sl, activation=$act"
                if $pn; then
                    python node_classification.py --gpu 7 --cuda --seed 42 --epochs 1000 --lr 0.01 --patience 20 --activation "$act" --self_loop "$sl" --sample_percent 0.90 --pairnorm --hidden_channels 128 --nb_hidlayers "$layer"
                else
                    python node_classification.py --gpu 7 --cuda --seed 42 --epochs 1000 --lr 0.01 --patience 20 --activation "$act" --self_loop "$sl" --sample_percent 0.90 --hidden_channels 128 --nb_hidlayers "$layer"
                fi
            done
        done
    done
done

```
在cora上node_classification结果为：
最佳参数

- layers: 4
- activation: relu
- pairnorm: False
- dropedge: 0.1
- selfloop: True

![image.png](https://cdn.nlark.com/yuque/0/2024/png/34306455/1704645697532-68645f01-e0b0-47b4-8742-9881a22c1a12.png#averageHue=%23fbfaf9&clientId=u82982991-226f-4&from=paste&height=467&id=u85c60541&originHeight=420&originWidth=565&originalType=binary&ratio=1&rotation=0&showTitle=false&size=27738&status=done&style=none&taskId=udf0a7f82-574a-4bf3-8f75-de126cba6d6&title=&width=627.77779440821)
在cora上link_prediction结果为：
最佳参数

- layers:2
- activation: relu
- pairnorm:false
- dropedge:0.2
- selfloop:false

![image.png](https://cdn.nlark.com/yuque/0/2024/png/34306455/1704645665405-687b97f9-bacc-428d-8656-32a2ac036e57.png#averageHue=%23fbfaf9&clientId=u82982991-226f-4&from=paste&height=486&id=u0a11636e&originHeight=437&originWidth=590&originalType=binary&ratio=1&rotation=0&showTitle=false&size=34117&status=done&style=none&taskId=u40ec5f6f-f588-4d41-8e14-e38de9d5d2a&title=&width=655.5555729218476)
在 Citeseer上node_classification结果为：
最佳参数

- layers:2
- activation:relu
- pairnorm:false
- dropedge:0.2
- selfloop:false

![image.png](https://cdn.nlark.com/yuque/0/2024/png/34306455/1704646267217-26e87d3a-bdc3-42c0-84e3-780b3222e1a6.png#averageHue=%23fbfafa&clientId=u82982991-226f-4&from=paste&height=469&id=ub63e2800&originHeight=422&originWidth=563&originalType=binary&ratio=1&rotation=0&showTitle=false&size=25778&status=done&style=none&taskId=ue1a4a443-16a8-443e-81af-055d7a2cd63&title=&width=625.5555721271189)
在 Citeseer上link_prediction结果为：
最佳参数

- layers:3
- activation: sigmoid
- pairnorm:false
- dropedge:0.1
- selfloop:true

![image.png](https://cdn.nlark.com/yuque/0/2024/png/34306455/1704646285539-44b1ef4c-ee7b-45ff-a3ea-5d419e8a12c4.png#averageHue=%23fbfaf9&clientId=u82982991-226f-4&from=paste&height=476&id=u7b71621e&originHeight=428&originWidth=583&originalType=binary&ratio=1&rotation=0&showTitle=false&size=32623&status=done&style=none&taskId=udb0b0446-8a81-4aed-a01e-b1f69e32d64&title=&width=647.777794938029)
在PPI上node_classification结果为：
最佳参数

- layers:3
- activation:tanh
- pairnorm:False
- dropedge:0.2
- selfloop:True

![ppi.png](https://cdn.nlark.com/yuque/0/2024/png/34306455/1704641365586-a9ec5224-83d5-4278-8a10-757cf19b5e62.png#averageHue=%23fcfbfa&clientId=u82982991-226f-4&from=ui&id=u52560191&originHeight=480&originWidth=640&originalType=binary&ratio=1&rotation=0&showTitle=false&size=21231&status=done&style=none&taskId=u56648939-cf3f-4824-8d90-210f0a4e8c5&title=)
在PPI上link_prediction结果为：
最佳参数

- layers:3
- activation: relu
- pairnorm:false
- dropedge:0.1
- selfloop:true

![image.png](https://cdn.nlark.com/yuque/0/2024/png/34306455/1704646363328-5d530cf2-d331-42d1-a3a2-3049377ea385.png#averageHue=%23fbfbfa&clientId=u82982991-226f-4&from=paste&height=468&id=u4e8d106e&originHeight=421&originWidth=584&originalType=binary&ratio=1&rotation=0&showTitle=false&size=24811&status=done&style=none&taskId=uddd928e3-93a2-4147-89dd-9e19c35e102&title=&width=648.8889060785746)
## 超参数分析
我们发现层数越高并不会一定能带来更好的实验效果，更高的layer可能会导致oversmoothing等问题；pairnorm效果在三个数据集的效果都不行
self-loop效果普遍会有提升；
dropedge可以很好的保持实验稳定性保持模型不会过拟合
## 实验收获

1. 利用 PyTorch 或 TensorFlow 的神经网络库，设计并实现了 GCN 模型
2. 在特定的图结构数据集上部署模型，完成了节点分类和链接预测任务
