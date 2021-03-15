##Given a list of PMIDS, return sentences with co-occurrence of BERN terms in abstract


import networkx as nx
import pymysql
import pandas as pd
import numpy as np

host="pubmed-database.c7xgknkzchxj.eu-west-2.rds.amazonaws.com"
port=3306
dbname="pubmed"
user="jks17"
password="password"



conn = pymysql.connect(host=host, user=user,port=port, passwd=password, db = dbname)

# bern[np.logical_and(bern['PMID'] == pmid, bern['Type'] != 1)]

def articles_to_knowledge(pmids,  host, port, dbname, user, password):
	abstracts = pd.read_sql('''SELECT PMID, AbstractText FROM A04_Abstract WHERE PMID in {}; '''.format(tuple(pmids)), con=conn)
	bern = pd.read_sql('''SELECT PMID, Mention, Type FROM B10_BERN_Main WHERE PMID in {}; '''.format(tuple(pmids)), con=conn)

	# import pdb; pdb.set_trace()
	#sentences
	dict_pmid_sentences = {}
	for pmid in pmids:
		if pmid not in abstracts['PMID'].tolist():
			continue
		sentences = [e.lower() for e in abstracts[abstracts['PMID'] ==pmid]['AbstractText'].values[0].split('.')]
		terms = [e.lower() for e in bern[np.logical_and(bern['PMID'] == pmid, bern['Type'] != '1')]['Mention'].tolist()]  #exclude type = 1 - species
		terms = list(set(terms))


		##Find co-ocurrences
		sentences_with_association = []
		for sentence in sentences:
			if sum([e in sentence for e in terms]) > 1:
				sentences_with_association.append(sentence)

		dict_pmid_sentences[pmid] = sentences_with_association

	return dict_pmid_sentences







