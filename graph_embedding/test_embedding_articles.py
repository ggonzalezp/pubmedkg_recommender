
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import numpy as np

import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


#Create dataset
dataset = torch.load('dataset/paper_paper_graph.pt')
data = dataset
data.edge_index  = to_undirected(data.edge_index)

#Split edges

from utils import train_test_split_edges
data = train_test_split_edges(data)




####to compute loss only on some node types
# out = model(...)
# loss = F.nll_loss(out[data.node_type == x], data.y[data.node_type == x])



from tensorboardX import SummaryWriter
display = SummaryWriter('.')

##############
#FROM EXAMPLES
##############

# from torch_geometric.utils import train_test_split_edges
# dataset = 'Cora'
# path = osp.join( 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]
# data.train_mask = data.val_mask = data.test_mask = data.y = None
# data = train_test_split_edges(data)




class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)
        self.conv2 = GCNConv(128,128)
        self.conv3 = GCNConv(128, 64)
        self.conv4 = GCNConv(64, 32)
    def encode(self):
        x = self.conv1(data.x, data.train_pos_edge_index)
        x = x.relu()
        x = self.conv2(x, data.train_pos_edge_index)
        x = x.relu()
        x = self.conv3(x, data.train_pos_edge_index)
        x = x.relu()
        return self.conv4(x, data.train_pos_edge_index)
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


device = torch.device('cuda', 2)
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    optimizer.zero_grad()
    z = model.encode()
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    link_probs = link_logits.sigmoid()
    rocauc=roc_auc_score(link_labels.detach().cpu().numpy(), link_probs.detach().cpu().numpy())
    return loss, rocauc


@torch.no_grad()
def test():
	model.eval()
	perfs = []
	for prefix in ["val", "test"]:
	    pos_edge_index = data[f'{prefix}_pos_edge_index']
	    neg_edge_index = data[f'{prefix}_neg_edge_index']
	    z = model.encode()
	    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
	    link_probs = link_logits.sigmoid()
	    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
	    perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
	return perfs


best_val_perf = test_perf = 0
for epoch in range(1, 10000):
    train_loss, train_perf = train()
    val_perf, tmp_test_perf = test()
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    log = 'Epoch: {:03d}, Loss: {:.4f}, Train ROCAUC: {:.4f}, Val ROCAUC: {:.4f}, Test ROCAUC: {:.4f}'
    display.add_scalars('data/loss', {
                'train_loss': train_loss,
                'val_roc_auc': val_perf,
                'test_roc_auc': tmp_test_perf
            }, epoch)
    print(log.format(epoch, train_loss, train_perf, val_perf, tmp_test_perf))



display.close()
z = model.encode()
final_edge_index = model.decode_all(z)



