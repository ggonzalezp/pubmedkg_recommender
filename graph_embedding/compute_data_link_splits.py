
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


#Create dataset split to evaluate performance on test set
data = torch.load('dataset/het_graph_paper_mesh.pk')

from utils import train_test_split_edges_relational
data = train_test_split_edges_relational(data, val_ratio=0.05, test_ratio=0.1)

torch.save(data, 'dataset/het_graph_paper_mesh_splitted_eval.pk')

#Create dataset split to train only on training set
data = torch.load('dataset/het_graph_paper_mesh.pk')

from utils import train_test_split_edges_relational
data = train_test_split_edges_relational(data, val_ratio=0, test_ratio=0)

torch.save(data, 'dataset/het_graph_paper_mesh_splitted.pk')