##Generates torch geometric dataset for each graph
#Only for the last 5 years!


import networkx as nx
import pandas as pd
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import numpy as np
from dask import dataframe as dd

from dask.distributed import Client
from torch_geometric.utils import to_undirected
import pickle


client = Client()



####################
###PAPER-PAPER GRAPH
####################

#Filter nodes and edges to keep only papers from 2015 onwards
papers = dd.read_csv('table_a01_articles.csv')
papers = papers[papers['Journal_JournalIssue_PubDate_Year'] > 2015]
papers = papers[['PMID']]
papers['PMID'] = papers['PMID'].astype(str)
papers = papers.compute()


edges = dd.read_csv('table_a14_reference_list.csv', low_memory=False, blocksize=128000000, dtype={'RefArticleId': 'object'}) #pendiente: subset
edges = edges.fillna(-1)
edges['PMID'] = edges['PMID'].astype(str)
edges['RefArticleId'] = edges['RefArticleId'].astype(str)
edges = client.persist(edges)


edges_ij_pmid = dd.merge(edges, papers, on = ["PMID"]).compute()

papers.columns = ['RefArticleId'] 
edges_final = dd.merge(edges_ij_pmid, papers, on=['RefArticleId'])
papers.columns = ['PMID'] 








#Paper nodes
papers = dd.read_csv('table_a01_articles.csv')
papers = papers[papers['Journal_JournalIssue_PubDate_Year'] > 2015]
papers['PMID'] = papers['PMID'].astype(str)
papers = papers.compute()
papers = papers.sort_values(axis=0, by='PMID', ascending=True)

#Reindexing dictionary
dict_reindex = dict(zip(papers['PMID'].tolist(), range(len(papers))))



#Save paper node information
papers_info = papers.copy()[['PMID', 'ArticleTitle']].reset_index(inplace=False, drop=True)
papers_info['node_index'] = [dict_reindex[e] for e in papers_info['PMID'].tolist()]


with open('paper_nodes_info.pickle', 'wb') as f:
	pickle.dump(papers_info, f)






#Input feature preprocessing

#Feature inputting
papers = papers.fillna(0)

#pre-processing of individual features
features = papers[['PMID', 'Language', 'Journal_ISSN', 'Journal_JournalIssue_Issue','Journal_JournalIssue_PubDate_Year']]
features['PMID'] = features['PMID'].astype(int)
#Encoding language
le = LabelEncoder()
features['Language'] = le.fit_transform(features['Language'])

#Formatting Journal ISSN
issn = [str(e).replace('-', '') for e in features['Journal_ISSN'].tolist()]
issn = [int(re.sub(r'[A-Z]+', '',e)) for e in issn]
features['Journal_ISSN'] = issn

#Journal issue volume
# features['Journal_JournalIssue_Volume'] = np.where(features['Journal_JournalIssue_Volume'] != 'Suppl', features['Journal_JournalIssue_Volume'], -1)
# jiv = [int(re.sub(r'[A-Z]', '', re.sub(r'[a-z]', '',str(e))).replace('(', '').replace(')', '').replace(' ', '').replace('-', '').replace('.', '').replace('*', '').replace(':', ''))
# 		if re.sub(r'[A-Z]', '', re.sub(r'[a-z]', '',str(e))).replace('(', '').replace(')', '').replace(' ', '').replace('-', '').replace('.', '').replace('*', '').replace(':', '') != '' else -1 for e in features['Journal_JournalIssue_Volume'].tolist()   ]

# 	split('(')[0].split(' ')[0].split('-')[0].split(')')[0].split('.')[-1]) if re.sub(r'[A-Z]', '', re.sub(r'[a-z]', '',str(e))).split('(')[0].split(' ')[0].split('-')[0].split(')')[0].split('.')[-1] != '' else 0 for e in features['Journal_JournalIssue_Volume'].tolist()   ]

# jiv = [int(re.sub(r'[A-Z]', '', re.sub(r'[a-z]', '',str(e))).split('(')[0].split(' ')[0].split('-')[0].split(')')[0].split('.')[-1]) if re.sub(r'[A-Z]', '', re.sub(r'[a-z]', '',str(e))).split('(')[0].split(' ')[0].split('-')[0].split(')')[0].split('.')[-1] != '' else 0 for e in features['Journal_JournalIssue_Volume'].tolist()   ]
# features['Journal_JournalIssue_Volume'] = jiv

#Journal issue

jii = []
for e in features['Journal_JournalIssue_Issue'].tolist():
	try:
		n = int(re.sub(r'[A-Z]', '', re.sub(r'[a-z]', '',str(e))).split('-')[0].split(' ')[0].split('(')[0].replace('_','').replace('.', '').replace(',','').replace('\"', '').replace('°', '').replace('`', ''))
	except:
		n = -1
	jii.append(n)


features['Journal_JournalIssue_Issue'] = jii

#Journal year
jy = [int(e) for e in features['Journal_JournalIssue_PubDate_Year']]
features['Journal_JournalIssue_PubDate_Year'] = jy



x_paper = torch.tensor(features.values, dtype=torch.float32)

#edge index paper-paper
edges_final['PMID'] = [dict_reindex[e] for e in edges_final['PMID'].tolist()]
edges_final['RefArticleId'] = [dict_reindex[e] for e in edges_final['RefArticleId'].tolist()]
edge_index_paper_paper = torch.LongTensor(edges_final.values.transpose())












####################
###PAPER-MESH GRAPH
####################
first_node_mesh_index = x_paper.size(0)

mesh = dd.read_csv('table_a06_mesh_heading_list.csv')
mesh = mesh[mesh['DescriptorName_MajorTopicYN'] == 'Y']
mesh['PMID'] = mesh['PMID'].astype(str)

mesh = mesh.compute()
mesh.index = mesh['PMID'].tolist()
mesh_filtered = dd.merge(mesh, papers, on = ["PMID"])
mesh_filtered = mesh_filtered.sort_values(axis=0, by='DescriptorName_UI', ascending=True)


#Get mesh node ids to create reindexing dictionary
descriptor_names = list(set(mesh_filtered['DescriptorName_UI'].tolist()))
dict_reindex_mesh = dict(zip(descriptor_names, range(first_node_mesh_index, first_node_mesh_index + len(descriptor_names))))


#Save mesh node information
mesh_info = pd.DataFrame([descriptor_names]).transpose()
mesh_info.columns=['DescriptorName_UI']
mesh_info['node_index'] = [dict_reindex_mesh[e] for e in mesh_info['DescriptorName_UI'].tolist()]
with open('mesh_nodes_info.pickle', 'wb') as f:
	pickle.dump(mesh_info, f)

####Node features Mesh nodes
x_mesh = torch.eye(len(set(mesh_filtered['DescriptorName_UI'])))


####Edges paper-mesh
mesh_filtered['PMID'] = [dict_reindex[e] for e in mesh_filtered['PMID'].tolist()]
mesh_filtered['DescriptorName_UI'] = [dict_reindex_mesh[e] for e in mesh_filtered['DescriptorName_UI'].tolist()]
edge_index_paper_mesh = torch.LongTensor(mesh_filtered[['PMID', 'DescriptorName_UI']].values.transpose())










#Create dataset
# edge_index_paper_paper = to_undirected(edge_index_paper_paper)
# edge_index_paper_mesh = to_undirected(edge_index_paper_mesh)

edge_type = torch.cat([torch.zeros(edge_index_paper_paper.size(1)), torch.ones(edge_index_paper_mesh.size(1))], 0)
edge_index = torch.cat([edge_index_paper_paper, edge_index_paper_mesh], 1)


dataset = Data(x_paper = x_paper, x_mesh=x_mesh, edge_index=edge_index, edge_type = edge_type, mesh_feature_dim = x_mesh.size(1), paper_feature_dim = x_paper.size(1))
torch.save(dataset, 'het_graph_paper_mesh.pk')






















