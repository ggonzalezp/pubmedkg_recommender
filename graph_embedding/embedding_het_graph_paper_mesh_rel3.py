##Includes relation-type specific matrix for decoding
#Computes negative sampling during trained (not pre-saved as in previous versions)


import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


#Create dataset
dataset = torch.load('dataset/het_graph_paper_mesh_splitted.pk')
data = dataset




from torch.nn import Linear
from torch_geometric.nn import RGCNConv



n_dim_initial_embedding = 16
n_relations = 2
num_bases = 4

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.transform_paper = Linear(data.paper_feature_dim, n_dim_initial_embedding)
		self.transform_mesh = Linear(data.mesh_feature_dim, n_dim_initial_embedding)
		self.conv1 = RGCNConv(n_dim_initial_embedding, 32, n_relations, num_bases=num_bases)
		self.conv2 = RGCNConv(32, 32, n_relations, num_bases=num_bases)
		self.conv3 = RGCNConv(32, 16, n_relations, num_bases=num_bases)
		self.decoding_matrix_paper_paper = Linear(16, 16)
		self.decoding_matrix_paper_mesh = Linear(16, 16)
	def encode(self):
		x = torch.cat([self.transform_paper(data.x_paper), self.transform_mesh(data.x_mesh)], 0)
		x = self.conv1(x, data.train_pos_edge_index, data.train_pos_edge_type)
		x = x.relu()
		x = self.conv2(x, data.train_pos_edge_index, data.train_pos_edge_type)
		x = x.relu()
		return self.conv3(x, data.train_pos_edge_index, data.train_pos_edge_type)
	def decode(self, z, pos_edge_index, pos_edge_type, neg_edge_index, neg_edge_type): #decoding with a relation-type specific decoding matrix --- assumming that the edges are in order (first all 0, then all 1)
		pos_0 = self.decoding_matrix_paper_paper(z[pos_edge_index[0]][pos_edge_type == 0]) * z[pos_edge_index[1]][pos_edge_type == 0]
		pos_1 = self.decoding_matrix_paper_mesh(z[pos_edge_index[0]][pos_edge_type == 1]) * z[pos_edge_index[1]][pos_edge_type == 1]
		neg_0 = self.decoding_matrix_paper_paper(z[neg_edge_index[0]][neg_edge_type == 0]) * z[neg_edge_index[1]][neg_edge_type == 0]
		neg_1 = self.decoding_matrix_paper_mesh(z[neg_edge_index[0]][neg_edge_type == 1]) * z[neg_edge_index[1]][neg_edge_type == 1]
		# edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
		# edge_type = torch.cat([pos_edge_type, neg_edge_type], dim=-1)
		logits = (torch.cat([pos_0, pos_1, neg_0, neg_1], 0)).sum(dim=-1)
		return logits
	def decode_all(self, z):
		prob_adj = z @ z.t()
		return (prob_adj > 0).nonzero(as_tuple=False).t()



device = torch.device('cuda', 7)
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)



from tensorboardX import SummaryWriter
display = SummaryWriter('.')


def get_link_labels(pos_edge_index, neg_edge_index):
	E = pos_edge_index.size(1) + neg_edge_index.size(1)
	link_labels = torch.zeros(E, dtype=torch.float, device=device)
	link_labels[:pos_edge_index.size(1)] = 1.
	return link_labels



def train():
	model.train()
	#negative sampling
	neg_row, neg_col = negative_sampling(data.train_pos_edge_index, 
																	num_nodes = data.num_nodes,  
																	num_neg_samples= data.train_pos_edge_index.size(1))
	to_keep = ~ torch.logical_and(neg_row >= data.x_paper.size(0) , neg_col >= data.x_paper.size(0)) #keep exclude mesh-mesh edges
	neg_row, neg_col = neg_row[to_keep], neg_col[to_keep]
	train_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
	train_neg_edge_type = torch.logical_or(torch.logical_and(neg_row < data.x_paper.size(0) , neg_col >= data.x_paper.size(0)), torch.logical_and(neg_row >= data.x_paper.size(0) , neg_col < data.x_paper.size(0))).to(torch.float32)
	sort_indices = torch.argsort(train_neg_edge_type)
	train_neg_edge_index = train_neg_edge_index[:, sort_indices]
	train_neg_edge_type = train_neg_edge_type[sort_indices]

	optimizer.zero_grad()

	z = model.encode()
	link_logits = model.decode(z, data.train_pos_edge_index, data.train_pos_edge_type, train_neg_edge_index, train_neg_edge_type)
	link_labels = get_link_labels(data.train_pos_edge_index, train_neg_edge_index)
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
		pos_edge_type = data[f'{prefix}_pos_edge_type']
		neg_edge_index = data[f'{prefix}_neg_edge_index']
		neg_edge_type = data[f'{prefix}_neg_edge_type']
		z = model.encode()
		link_logits = model.decode(z, pos_edge_index, pos_edge_type, neg_edge_index, neg_edge_type)
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






# self.conv1 = RGCNConv(n_dim_initial_embedding, 64, n_relations, num_bases=num_bases)
# self.conv2 = RGCNConv(64, 64, n_relations, num_bases=num_bases)
# self.conv3 = RGCNConv(64, 16, n_relations, num_bases=num_bases)


display.close()
