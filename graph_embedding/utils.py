import math
import torch
from torch_geometric.utils import to_undirected, negative_sampling
import scipy.sparse as sp
import random

#Adapted from torch_geometric to delete the mask creation
#https://pytorch-geometric.readthedocs.io/en/1.6.3/_modules/torch_geometric/utils/train_test_split_edges.html 
def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row_orig, col_orig = data.edge_index
    row, col = data.edge_index
    # data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)


    ## to counteract high dimensionality
    #Negative edges
    neg_row, neg_col = negative_sampling(data.edge_index, num_nodes = num_nodes, num_neg_samples=n_v+n_t, force_undirected=False)


    # neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    # neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    # neg_adj_mask[row, col] = 0

    # neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    # perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    # neg_row, neg_col = neg_row[perm], neg_col[perm]

    # neg_adj_mask[neg_row, neg_col] = 0
    # data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    #remove original edge index
    data.edge_index = None

    return data



#Adapted from torch_geometric to delete the mask creation and to support sampling edge-type dependent
#https://pytorch-geometric.readthedocs.io/en/1.6.3/_modules/torch_geometric/utils/train_test_split_edges.html 
def train_test_split_edges_relational(data, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.


    #######
    #Type 0 - paper - paper
    #######
    num_papers = data.x_paper.size(0)
    edge_index_papers = data.edge_index[:, data.edge_type == 0]
    row, col = edge_index_papers

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    val_pos_edge_index_paper_paper = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    test_pos_edge_index_paper_paper = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    train_pos_edge_index_paper_paper = torch.stack([r, c], dim=0)
    train_pos_edge_index_paper_paper = to_undirected(train_pos_edge_index_paper_paper)

    #Negative edges
    neg_row, neg_col = negative_sampling(edge_index_papers, num_nodes = num_papers, num_neg_samples=n_v+n_t, force_undirected=False)

    row, col = neg_row[:n_v], neg_col[:n_v]
    val_neg_edge_index_paper_paper = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    test_neg_edge_index_paper_paper = torch.stack([row, col], dim=0)


    #######
    #Type 1 - paper - mesh
    #######
    num_mesh = data.x_mesh.size(0)
    edge_index_mesh = data.edge_index[:, data.edge_type == 1]
    row, col = edge_index_mesh


    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    val_pos_edge_index_paper_mesh = torch.stack([r, c], dim=0)

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    test_pos_edge_index_paper_mesh = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    train_pos_edge_index_paper_mesh = torch.stack([r, c], dim=0)
    train_pos_edge_index_paper_mesh = to_undirected(train_pos_edge_index_paper_mesh)




    #Negative edges
    neg_row, neg_col = negative_sampling(edge_index_mesh, num_nodes = num_papers + num_mesh,  num_neg_samples=110*(n_v+n_t), force_undirected=False)
    to_keep = torch.logical_or(torch.logical_and(neg_row < num_papers , neg_col >= num_papers), torch.logical_and(neg_row >= num_papers , neg_col < num_papers)) #keep only nodes of type 1
    neg_row, neg_col = neg_row[to_keep], neg_col[to_keep]


    perm = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_t], neg_col[:n_t] 
    test_neg_edge_index_paper_mesh = torch.stack([row, col], dim=0)    

    row, col = neg_row[n_t:], neg_col[n_t:]                         #I don't have the full # for validation set 
    val_neg_edge_index_paper_mesh = torch.stack([row, col], dim=0)

    ### 
    ##Joining edges
    ###
    data.train_pos_edge_index = torch.cat([train_pos_edge_index_paper_paper, train_pos_edge_index_paper_mesh], 1)
    data.train_pos_edge_type = torch.cat([torch.zeros(train_pos_edge_index_paper_paper.size(1)), torch.ones(train_pos_edge_index_paper_mesh.size(1))], 0)

    data.val_pos_edge_index = torch.cat([val_pos_edge_index_paper_paper, val_pos_edge_index_paper_mesh], 1)
    data.val_pos_edge_type = torch.cat([torch.zeros(val_pos_edge_index_paper_paper.size(1)), torch.ones(val_pos_edge_index_paper_mesh.size(1))], 0)

    data.val_neg_edge_index = torch.cat([val_neg_edge_index_paper_paper, val_neg_edge_index_paper_mesh], 1)
    data.val_neg_edge_type = torch.cat([torch.zeros(val_neg_edge_index_paper_paper.size(1)), torch.ones(val_neg_edge_index_paper_mesh.size(1))], 0)

    data.test_pos_edge_index = torch.cat([test_pos_edge_index_paper_paper, test_pos_edge_index_paper_mesh], 1)
    data.test_pos_edge_type = torch.cat([torch.zeros(test_pos_edge_index_paper_paper.size(1)), torch.ones(test_pos_edge_index_paper_mesh.size(1))], 0)


    data.test_neg_edge_index = torch.cat([test_neg_edge_index_paper_paper, test_neg_edge_index_paper_mesh], 1)
    data.test_neg_edge_type = torch.cat([torch.zeros(test_neg_edge_index_paper_paper.size(1)), torch.ones(test_neg_edge_index_paper_mesh.size(1))], 0)

    # #remove original types
    # data.edge_type = None
    # data.edge_index = None


    ##Extra - sampling negative edges for training - did this during training before, but it's too slow to do on each epoch
    neg_row, neg_col = negative_sampling(data.train_pos_edge_index, 
                                        num_nodes = data.num_nodes,  
                                        num_neg_samples= data.train_pos_edge_index.size(1))
    to_keep = ~ torch.logical_and(neg_row >= num_papers , neg_col >= num_papers) #keep exclude mesh-mesh edges
    neg_row, neg_col = neg_row[to_keep], neg_col[to_keep]
    data.train_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
    data.train_neg_edge_type = torch.logical_or(torch.logical_and(neg_row < num_papers , neg_col >= num_papers), torch.logical_and(neg_row >= num_papers , neg_col < num_papers)).to(torch.float32)
    sort_indices = torch.argsort(data.train_neg_edge_type)
    data.train_neg_edge_index = data.train_neg_edge_index[:, sort_indices]
    data.train_neg_edge_type = data.train_neg_edge_type[sort_indices]

    return data




