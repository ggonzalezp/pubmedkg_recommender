import pickle
import torch
import os.path as osp
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np



def get_embeddings_to_visualize(descriptors, dict_pmid_count_mesh):
	
	
	#Retrieves PMIDs with descriptors
	pmids_with_mesh = []
	base_path = '../../../graph_embedding/dataset/dict_mesh_pmids'
	for mesh in descriptors:
	    if osp.isfile(osp.join(base_path, mesh + '.json')):
	        pmids_with_mesh += json.load(open(osp.join(base_path, '{}.json'.format(mesh)), 'r'))


	pmids_with_mesh = list(set(pmids_with_mesh))
	pmids_with_mesh = pd.DataFrame.from_dict(dict_pmid_count_mesh, orient='index')[pd.DataFrame.from_dict(dict_pmid_count_mesh, orient='index')[0] >= 0.5 * len(descriptors)].index.tolist() #filtering by # of mesh terms


	#Loads embeddings
	embeddings = []
	for pmid in pmids_with_mesh:
			with open('../../../graph_embedding/out_embeddings/processed/{}.pickle'.format(pmid), 'rb') as f:
				embeddings.append(pickle.load(f))

	for d in descriptors:
		with open('../../../graph_embedding/out_embeddings/processed/{}.pickle'.format(d), 'rb') as f:
			embeddings.append(pickle.load(f))

	# import pdb; pdb.set_trace()
	#Embeddings
	embeddings = pd.DataFrame(embeddings)
	mask = [i for i in range(16)]
	emb = pd.DataFrame(TSNE(n_components=2).fit_transform(np.array(embeddings[mask])))
	# import pdb; pdb.set_trace()
	emb = pd.concat([emb, embeddings[[16, 17]]], 1)
	emb.columns = ['x', 'y', 'Name', 'Node type']


	return emb


