##Post-process embeddings 
#Creates dictionary with mesh_id: paper_pmids (top neighbors using threshold)
#Creates dictionary with pmid: paper_pmids (top neighbors using threshold)


import torch
import pickle





#Loads data
with open('dataset/mesh_nodes_info.pickle', 'rb') as f:
    mesh_info = pickle.load(f)



with open('dataset/paper_nodes_info.pickle', 'rb') as f:
    papers_info = pickle.load(f)


data = torch.load('dataset/het_graph_paper_mesh_splitted.pk')


device = torch.device('cuda', 7)
embeddings = torch.load('paper_mesh_embeddings.pk', map_location=device)
decoding_matrix_paper_paper = torch.load('decoding_matrix_paper_paper.pk', map_location=device)
decoding_matrix_paper_mesh = torch.load('decoding_matrix_paper_mesh.pk', map_location=device)



#Create dictionary node_idx: PMID , and node_idx:descriptorname
dict_mesh_indexer = dict(zip(mesh_info['node_index'], mesh_info['DescriptorName_UI']))
dict_paper_indexer = dict(zip(papers_info['node_index'], papers_info['PMID']))



paper_embeddings = embeddings[0:len(papers_info), :]
mesh_embeddings = embeddings[len(papers_info):, :]


########Find top neighbors of mesh nodes and create a dictionary with mesh: [pmids]
k = 50
topk = []
#Decoding paper-mesh : dot product of (emb_mesh * decoding_matrix) and emb_mesh
for i in range(100, len(mesh_info), 100):
	mesh_to_paper = torch.matmul(decoding_matrix_paper_mesh(mesh_embeddings[i-100:i, ]).detach(), paper_embeddings.t())
	topk.append(torch.argsort(mesh_to_paper, -1, descending=True)[:, :k].detach().cpu())

mesh_to_paper = torch.matmul(decoding_matrix_paper_mesh(mesh_embeddings[i:, ]).detach(), paper_embeddings.t())
topk.append(torch.argsort(mesh_to_paper, -1, descending=True)[:, :k].detach().cpu())

topk = torch.cat(topk)


#Create dictionary with mesh_id : pmids
num_papers = len(paper_embeddings)
dict_mesh_id_paper_pmid = {}
for i in range(len(topk)):
	entry = topk[i, :].detach().cpu().numpy().tolist()
	dict_mesh_id_paper_pmid[dict_mesh_indexer[i+num_papers]] = [dict_paper_indexer[e] for e in entry]



########Find top neighbors of paper nodes and create a dictionary with pmid: [pmids]
from dask import dataframe as dd
from dask.distributed import Client
client = Client()

z_1 = decoding_matrix_paper_paper(paper_embeddings).detach().cpu().numpy()
z_2 = paper_embeddings.t().detach().cpu().numpy()
z_1 = dd.from_array(z_1)
z_2 = dd.from_array(z_2)

z = dd.tensordot(z_1, z_2)





k = 50
topk = []
#Decoding paper-paper : dot product of (emb_mesh * decoding_matrix) and emb_mesh
for i in range(100, len(papers_info), 100):
	z_1 = decoding_matrix_paper_paper(paper_embeddings).detach().cpu().numpy()
	z_2 = paper_embeddings.t().detach().cpu().numpy()
	z_1 = dd.from_array(z_1)
	z_2 = dd.from_array(z_2)

	z = dd.tensordot(z_1, z_2)

	paper_to_paper = torch.matmul(decoding_matrix_paper_paper(paper_embeddings[i-100:i, ]).detach(), paper_embeddings.t())



	quant_thresholds = torch.quantile(paper_to_paper, q = 0.75, dim = -1).unsqueeze(1)
	indices_row, indices_col = torch.where(paper_to_paper > quant_thresholds)
for i in indices_row:
	topk.append(indices_col[indices_row == i].detach().cpu().numpy().tolist())

paper_to_paper = torch.matmul(decoding_matrix_paper_paper(paper_embeddings[i:, ]).detach(), paper_embeddings.t())
topk.append(torch.argsort(paper_to_paper, -1, descending=True)[:, :k].detach().cpu())

topk = torch.cat(topk)


#Create dictionary with pmid : pmids
dict_mesh_id_paper_pmid = {}
for i in range(len(topk)):
	entry = topk[i, :].detach().cpu().numpy().tolist()
	dict_mesh_id_paper_pmid[dict_paper_indexer[i]] = [dict_paper_indexer[e] for e in entry]




















