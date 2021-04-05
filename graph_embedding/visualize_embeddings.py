#code to visualize embeddings
import pickle
import torch
import os.path as osp
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import os
import json

from sklearn.manifold import TSNE


with open('dataset/mesh_nodes_info.pickle', 'rb') as f:
    mesh_info = pickle.load(f)


mesh_info.index = mesh_info['DescriptorName_UI'].tolist()


with open('dataset/paper_nodes_info.pickle', 'rb') as f:
    papers_info = pickle.load(f)



papers_info.index = papers_info['PMID'].tolist()


# #Loads embeddings
device = torch.device('cpu')
ids_of_embeddings = papers_info['PMID'].tolist() + mesh_info['DescriptorName_UI'].tolist()

embeddings = torch.load('out_embeddings/paper_mesh_embeddings.pk', map_location=device)
embeddings = pd.DataFrame(embeddings.numpy())
embeddings.index = ids_of_embeddings
embeddings['label'] = papers_info['ArticleTitle'].tolist() + mesh_info['name'].tolist()
embeddings['Node type'] = ['Paper' for i in range(len(papers_info))] + ['Keyword' for i in range(len(mesh_info))]


##Saves processed embeddings
outdir = 'out_embeddings/processed'
os.makedirs(outdir, exist_ok=True)
for id in embeddings.index.tolist():
	with open(osp.join(outdir,'{}.pickle'.format(id)), 'wb') as f:
		pickle.dump(embeddings.loc[id].tolist(), f)
