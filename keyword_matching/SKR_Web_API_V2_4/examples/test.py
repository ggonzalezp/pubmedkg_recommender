###Calling it from command-line
from subprocess import Popen, PIPE
import pandas as pd
import pymysql
import numpy as np
import networkx as nx
import simplejson as json
import pickle
import json
from collections import Counter
from glob import glob
import os.path as osp
import time

def run_get_graph():
    # import pdb; pdb.set_trace()
    p = Popen(['../run.sh', 'GenericBatchUser', 'test.txt'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    output = str(output)
    host="pubmed-database.c7xgknkzchxj.eu-west-2.rds.amazonaws.com"
    port=3306
    dbname="pubmed"
    user="jks17"
    password="password"

    conn = pymysql.connect(host=host, user=user,port=port,
                                   passwd=password, db = dbname)
    # get id of the keywords
    # import pdb; pdb.set_trace()
    descriptors = []
    for entry in output.split('\\n'):
        if entry != '\'':
            if 'MM' in entry.split('|')[7]:
                descriptors.append(entry.split('|')[2])

    print('got descriptors')

    edges = pd.read_sql('''
        SELECT DISTINCT t.PMID, t.RefArticleId
        FROM A14_ReferenceList t
        JOIN A06_MeshHeadingList t1 ON t1.PMID = t.PMID
        JOIN A06_MeshHeadingList t2 ON t2.PMID = t.RefArticleId
        WHERE t1.DescriptorName_UI in {}
        AND  t2.DescriptorName_UI in {}
        LIMIT 50;
        '''.format(tuple(descriptors), tuple(descriptors)), con=conn)
    G=nx.Graph()
    for arr in edges.values.astype('int'):
        G.add_edge(arr[0],arr[1])

    

    conn.close()

    nx.write_gexf( G , 'test_graph.gexf' )

    return G, descriptors

def run_get_graph_2(): #using dictionaries with pre-processed data to facilitate querying
    t0 = time.time()
    
    p = Popen(['../run.sh', 'GenericBatchUser', 'test.txt'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    output = str(output)
    
    # get id of the keywords
    # import pdb; pdb.set_trace()
    descriptors = []
    for entry in output.split('\\n'):
        if entry != '\'':
            if 'MM' in entry.split('|')[7]:
                descriptors.append(entry.split('|')[2])
    t1 = time.time()
    print('got descriptors. Time: {:.2f} secs'.format(t1-t0))

    t0 = time.time()
    
    base_path = '../../../graph_embedding/dataset/dict_mesh_pmids'
    # paths = sorted(glob(base_path+'/*'))
    
    #Get PMIDs with ALL mesh terms in input
    # import pdb; pdb.set_trace()
    pmids_with_mesh = []
    for mesh in descriptors:
        if osp.isfile(osp.join(base_path, mesh + '.json')):
            pmids_with_mesh += json.load(open(osp.join(base_path, '{}.json'.format(mesh)), 'r'))


    # pmids_with_mesh = [dict_mesh_pmids[e] if e in dict_mesh_pmids else '' for e in descriptors]
    # pmids_with_mesh = list(set(pmids_with_mesh[0]).union(*pmids_with_mesh[1:]))
    dict_pmid_count_mesh = Counter(pmids_with_mesh)
    # pmids_with_mesh = list(set(pmids_with_mesh)) #without filtering
    pmids_with_mesh = pd.DataFrame.from_dict(dict_pmid_count_mesh, orient='index')[pd.DataFrame.from_dict(dict_pmid_count_mesh, orient='index')[0] >= 0.5 * len(descriptors)].index.tolist() #filtering by # of mesh terms

    #Get the edges
    base_path = '../../../graph_embedding/dataset/dict_pmid_refs'
    # paths = sorted(glob(base_path+'/*'))
    # import pdb; pdb.set_trace()
    tuples_edges = []
    i = 1
    for pmid in pmids_with_mesh:
        if osp.isfile(osp.join(base_path, pmid + '.json')):
            inters = set(json.load(open(osp.join(base_path, '{}.json'.format(pmid)), 'r'))).intersection(pmids_with_mesh)
            if len(inters) == 0:
                 continue
            else:
                pmids_refs = list(inters)
                for e in pmids_refs:
                    tuples_edges.append((pmid, e))
        print('{}/{}'.format(i, len(pmids_with_mesh)))
        i += 1


    G=nx.Graph()
    for arr in tuples_edges:
        G.add_edge(int(arr[0]),int(arr[1]))

    t1 = time.time()
    print('Created graph. Time: {:.2f} secs'.format(t1 - t0))
    nx.write_gexf( G , 'test_graph.gexf' )

    return G, descriptors, dict_pmid_count_mesh

def run_get_graph_using_embeddings(mesh_info, embeddings):
    # import pdb; pdb.set_trace()
    p = Popen(['../run.sh', 'GenericBatchUser', 'test.txt'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    output = str(output)

    # get id of the keywords
    # import pdb; pdb.set_trace()
    descriptors = []
    for entry in output.split('\\n'):
        if entry != '\'':
            if 'MM' in entry.split('|')[7]:
                descriptors.append(entry.split('|')[2])

    print('got descriptors')





    host="pubmed-database.c7xgknkzchxj.eu-west-2.rds.amazonaws.com"
    port=3306
    dbname="pubmed"
    user="jks17"
    password="password"

    conn = pymysql.connect(host=host, user=user,port=port,
                                   passwd=password, db = dbname)

    edges = pd.read_sql('''
        SELECT DISTINCT t.PMID, t.RefArticleId
        FROM A14_ReferenceList t
        JOIN A06_MeshHeadingList t1 ON t1.PMID = t.PMID
        JOIN A06_MeshHeadingList t2 ON t2.PMID = t.RefArticleId
        WHERE t1.DescriptorName_UI in {}
        AND  t2.DescriptorName_UI in {}
        LIMIT 50;
        '''.format(tuple(descriptors), tuple(descriptors)), con=conn)
    G=nx.Graph()
    for arr in edges.values.astype('int'):
        G.add_edge(arr[0],arr[1])

    

    conn.close()

    nx.write_gexf( G , 'test_graph.gexf' )

    return G, descriptors