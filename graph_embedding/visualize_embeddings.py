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


#Loads embeddings
device = torch.device('cpu')
embeddings = torch.load('out_embeddings/paper_mesh_embeddings.pk', map_location=device)


descriptors = ['D002318', 'D015415']

pmids_with_mesh = []
base_path = 'dataset/dict_mesh_pmids'
for mesh in descriptors:
    if osp.isfile(osp.join(base_path, mesh + '.json')):
        pmids_with_mesh += json.load(open(osp.join(base_path, '{}.json'.format(mesh)), 'r'))


pmids_with_mesh = list(set(pmids_with_mesh))


ids_of_embeddings = papers_info['PMID'].tolist() + mesh_info['DescriptorName_UI'].tolist()


#Standard scaling
scaler = StandardScaler()
embeddings = pd.DataFrame(embeddings.numpy())
# embeddings = pd.DataFrame(embeddings.numpy())
embeddings.index = ids_of_embeddings




#Embeddings of papers
emb_papers = embeddings.loc[pmids_with_mesh]


#Embeddings of mesh
emb_mesh = embeddings.loc[descriptors]



emb = pd.concat([emb_papers, emb_mesh], 0)
emb = pd.DataFrame(TSNE(n_components=2).fit_transform(emb))
emb['label'] = papers_info.loc[emb_papers.index]['ArticleTitle'].tolist() + mesh_info.loc[emb_mesh.index]['name'].tolist()
emb.columns = ['x', 'y', 'Name']
emb['Node type'] = ['Paper' for i in range(len(emb_papers))] + ['Keyword' for i in range(len(emb_mesh))]




import plotly.express as px

fig = px.scatter(emb,
    x='x',
    y='y',
    color='Node type',
    opacity=0.8,
	hover_data={'x':False,
				'y':False,
				'Name': True
	}
)


fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Roboto"
    )
)

fig.update_traces(marker=dict(size=30),
                  selector=dict(mode='markers'))

fig.write_html("figure.html")


##Saves processed embeddings
embeddings = torch.load('out_embeddings/paper_mesh_embeddings.pk', map_location=device)
# embeddings = embeddings[:, 0:2]
embeddings = pd.DataFrame(embeddings.numpy())
embeddings.index = ids_of_embeddings

embeddings['label'] = papers_info['ArticleTitle'].tolist() + mesh_info['name'].tolist()
# embeddings.columns = ['x', 'y', 'Name']
embeddings['Node type'] = ['Paper' for i in range(len(papers_info))] + ['Keyword' for i in range(len(mesh_info))]


# torch.save(embeddings, 'out_embeddings/paper_mesh_embeddings_processed.pk')

# with open('out_embeddings/paper_mesh_embeddings_processed.pickle', 'wb') as f:
# 	pickle.dump(embeddings, f)

outdir = 'out_embeddings/processed'
os.makedirs(outdir, exist_ok=True)
for id in embeddings.index.tolist():
	with open(osp.join(outdir,'{}.pickle'.format(id)), 'wb') as f:
		pickle.dump(embeddings.loc[id].tolist(), f)
