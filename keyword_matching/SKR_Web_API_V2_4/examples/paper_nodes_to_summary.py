from transformers import pipeline
import networkx as nx
import pymysql
import pandas as pd
import numpy as np

def articles_to_summary(pmids,  host, port, dbname, user, password, summarizer):
    conn = pymysql.connect(host=host, user=user,port=port, passwd=password, db = dbname)
    abstracts = pd.read_sql('''SELECT PMID, AbstractText FROM A04_Abstract WHERE PMID in {}; '''.format(tuple(pmids)), con=conn)
    dict_pmid_sentences = {}
    for pmid in pmids:
        if int(pmid) not in abstracts['PMID'].tolist():
            dict_pmid_sentences[pmid] = ''
            continue
        abstract = abstracts[abstracts['PMID'] ==int(pmid)]['AbstractText'].values[0]
        if abstract:
            summary = summarizer(abstract, min_length=1, max_length=50)
            summary = summary[0]['summary_text']
        else:
            summary = ''
        dict_pmid_sentences[pmid] = summary
        
    return dict_pmid_sentences


