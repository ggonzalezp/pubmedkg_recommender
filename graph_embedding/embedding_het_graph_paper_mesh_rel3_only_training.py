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
data.num_nodes = data.x_paper.size(0) + data.x_mesh.size(0)



from torch.nn import Linear
from torch_geometric.nn import RGCNConv



n_dim_initial_embedding = 16
n_relations = 2
num_bases = 4
hidden_dim = 128
embedding_dim = 16
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.transform_paper = Linear(data.paper_feature_dim, n_dim_initial_embedding)
		self.transform_mesh = Linear(data.mesh_feature_dim, n_dim_initial_embedding)
		self.conv1 = RGCNConv(n_dim_initial_embedding, 32, n_relations, num_bases=num_bases)
		self.conv2 = RGCNConv(32, 32, n_relations, num_bases=num_bases)
		self.conv3 = RGCNConv(32, 32, n_relations, num_bases=num_bases)
		self.conv4 = RGCNConv(32, embedding_dim, n_relations, num_bases=num_bases)
		self.decoding_matrix_paper_paper = Linear(embedding_dim, embedding_dim)
		self.decoding_matrix_paper_mesh = Linear(embedding_dim, embedding_dim)
	def encode(self):
		x = torch.cat([self.transform_paper(data.x_paper), self.transform_mesh(data.x_mesh)], 0)
		x = self.conv1(x, data.train_pos_edge_index, data.train_pos_edge_type)
		x = x.relu()
		x = self.conv2(x, data.train_pos_edge_index, data.train_pos_edge_type)
		x = x.relu()
		x = self.conv3(x, data.train_pos_edge_index, data.train_pos_edge_type)
		x = x.relu()
		return self.conv4(x, data.train_pos_edge_index, data.train_pos_edge_type)
	def decode(self, z, pos_edge_index, pos_edge_type, neg_edge_index, neg_edge_type): #decoding with a relation-type specific decoding matrix --- assumming that the edges are in order (first all 0, then all 1)
		pos_0 = self.decoding_matrix_paper_paper(z[pos_edge_index[0]][pos_edge_type == 0]) * z[pos_edge_index[1]][pos_edge_type == 0]
		pos_1 = self.decoding_matrix_paper_mesh(z[pos_edge_index[0]][pos_edge_type == 1]) * z[pos_edge_index[1]][pos_edge_type == 1]
		neg_0 = self.decoding_matrix_paper_paper(z[neg_edge_index[0]][neg_edge_type == 0]) * z[neg_edge_index[1]][neg_edge_type == 0]
		neg_1 = self.decoding_matrix_paper_mesh(z[neg_edge_index[0]][neg_edge_type == 1]) * z[neg_edge_index[1]][neg_edge_type == 1]
		# edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
		# edge_type = torch.cat([pos_edge_type, neg_edge_type], dim=-1)
		logits = (torch.cat([pos_0, pos_1, neg_0, neg_1], 0)).sum(dim=-1)
		return logits
	# def decode_all(self, z):
	# 	prob_adj = z @ z.t()
	# 	return (prob_adj > 0).nonzero(as_tuple=False).t()



device = torch.device('cuda', 7)
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)



from tensorboardX import SummaryWriter
# display = SummaryWriter('.')


def get_link_labels(pos_edge_index, neg_edge_index):
	E = pos_edge_index.size(1) + neg_edge_index.size(1)
	link_labels = torch.zeros(E, dtype=torch.float, device=device)
	link_labels[:pos_edge_index.size(1)] = 1.
	return link_labels



def train():
	alpha = 0.7
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
	link_logits_paper_paper = model.decode(z, data.train_pos_edge_index[:, data.train_pos_edge_type == 0], data.train_pos_edge_type[data.train_pos_edge_type == 0], train_neg_edge_index[:, train_neg_edge_type ==0], train_neg_edge_type[train_neg_edge_type ==0])
	link_logits_paper_mesh = model.decode(z,  data.train_pos_edge_index[:, data.train_pos_edge_type == 1], data.train_pos_edge_type[data.train_pos_edge_type == 1], train_neg_edge_index[:, train_neg_edge_type ==1], train_neg_edge_type[train_neg_edge_type ==1])
	link_labels_paper_paper = get_link_labels(data.train_pos_edge_index[:, data.train_pos_edge_type == 0], train_neg_edge_index[:, train_neg_edge_type ==0])
	link_labels_paper_mesh = get_link_labels(data.train_pos_edge_index[:, data.train_pos_edge_type == 1], train_neg_edge_index[:, train_neg_edge_type ==1])
	loss_paper_paper = F.binary_cross_entropy_with_logits(link_logits_paper_paper, link_labels_paper_paper)
	loss_paper_mesh = F.binary_cross_entropy_with_logits(link_logits_paper_mesh, link_labels_paper_mesh)
	loss = (1/2) * ((1 - alpha) * loss_paper_paper + alpha * loss_paper_mesh)
	# loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
	loss.backward()
	optimizer.step()
	link_probs = link_logits.sigmoid()
	link_probs_paper_paper = link_logits_paper_paper.sigmoid()
	link_probs_paper_mesh = link_logits_paper_mesh.sigmoid()
	rocauc=roc_auc_score(link_labels.detach().cpu().numpy(), link_probs.detach().cpu().numpy())
	roc_auc_pp=roc_auc_score(link_labels_paper_paper.detach().cpu().numpy(), link_probs_paper_paper.detach().cpu().numpy())
	roc_auc_pm=roc_auc_score(link_labels_paper_mesh.detach().cpu().numpy(), link_probs_paper_mesh.detach().cpu().numpy())
	return loss, rocauc, roc_auc_pp, roc_auc_pm




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

		link_logits_paper_paper = model.decode(z, pos_edge_index[:, pos_edge_type == 0], pos_edge_type[pos_edge_type == 0], neg_edge_index[:, neg_edge_type ==0], neg_edge_type[neg_edge_type ==0])
		link_logits_paper_mesh = model.decode(z,  pos_edge_index[:, pos_edge_type == 1], pos_edge_type[pos_edge_type == 1], neg_edge_index[:, neg_edge_type ==1], neg_edge_type[neg_edge_type ==1])
		link_probs_paper_paper = link_logits_paper_paper.sigmoid()
		link_probs_paper_mesh = link_logits_paper_mesh.sigmoid()

		link_labels_paper_paper = get_link_labels(pos_edge_index[:, pos_edge_type == 0], neg_edge_index[:, neg_edge_type ==0])
		link_labels_paper_mesh = get_link_labels(pos_edge_index[:, pos_edge_type == 1], neg_edge_index[:, neg_edge_type ==1])

		perfs.append(roc_auc_score(link_labels_paper_paper.cpu(), link_probs_paper_paper.cpu()))
		perfs.append(roc_auc_score(link_labels_paper_mesh.cpu(), link_probs_paper_mesh.cpu()))
	return perfs


outdir = 'out_embeddings'
os.makedirs(outdir, exist_ok = True)

best_perf = best_pp = best_pm = 0
for epoch in range(1, 500):
	train_loss, train_perf, train_perf_pp, train_perf_pm = train()
	if train_perf > best_perf:
		best_perf = train_perf
		best_pp = train_perf_pp
		best_pm = train_perf_pm
		torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },
            os.path.join(outdir, 'best_model.pt'))
	log = 'Epoch: {:03d}, Loss: {:.4f}, Train ROCAUC: {:.4f}, paper-paper: {:.4f}, paper-mesh: {:.4f}'
		# display.add_scalars('data/loss', {
		# 									'train_loss': train_loss,
		# 									'val_roc_auc': val_perf,
		# 									'test_roc_auc': tmp_test_perf
		# 									}, epoch)
	print(log.format(epoch, train_loss, train_perf, train_perf_pp, train_perf_pm ))






# self.conv1 = RGCNConv(n_dim_initial_embedding, 64, n_relations, num_bases=num_bases)
# self.conv2 = RGCNConv(64, 64, n_relations, num_bases=num_bases)
# self.conv3 = RGCNConv(64, 16, n_relations, num_bases=num_bases)
print('Best train ROCAUC: {:.4f}, paper-paper: {:.4f}, paper-mesh: {:.4f}'.format(best_perf, best_pp, best_pm))
checkpoint = torch.load(os.path.join(outdir, 'best_model.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
embeddings = model.encode()
torch.save(embeddings.detach(), osp.join(outdir, 'paper_mesh_embeddings.pk'))
torch.save(model.decoding_matrix_paper_paper, osp.join(outdir, 'decoding_matrix_paper_paper.pk'))
torch.save(model.decoding_matrix_paper_mesh, osp.join(outdir, 'decoding_matrix_paper_mesh.pk'))


# display.close()
##best one Best train ROCAUC: 0.8836, paper-paper: 0.8679, paper-mesh: 0.5909