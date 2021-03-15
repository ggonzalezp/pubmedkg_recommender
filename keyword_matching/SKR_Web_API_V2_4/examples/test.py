###Calling it from command-line
from subprocess import Popen, PIPE
import pandas as pd
import pymysql
import numpy as np
import networkx as nx
import simplejson as json

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


##to create A14 table    
# CREATE TABLE A14_ReferenceList_20 AS SELECT * FROM A14_ReferenceList WHERE PMID IN (SELECT PMID FROM A01_Articles_20) AND RefArticleId IN (SELECT PMID FROM A01_Articles_20) ;