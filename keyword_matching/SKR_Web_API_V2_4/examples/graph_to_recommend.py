import networkx as nx
import pymysql
import pandas as pd

k_papers = 5 # how many papers to return
k_people = 5 # how many people to return
host="pubmed-database.c7xgknkzchxj.eu-west-2.rds.amazonaws.com"
port=3306
dbname="pubmed"
user="jks17"
password="password"

def graph_to_recommend(graph, host, port, dbname, user, password):
    pagerank = nx.pagerank(graph)
    pagerank_ordered = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1], reverse = True)}
    top_k_papers_pmids = list(pagerank_ordered.keys())[:k_papers]
    # now get the paper titles of these 5 papers

    conn = pymysql.connect(host=host, user=user,port=port, passwd=password, db = dbname)

    articles = pd.read_sql('''SELECT ArticleTitle FROM A01_Articles WHERE PMID in {}; '''.format(tuple(top_k_papers_pmids)), con=conn)

    top_k_papers = [a[0] for a in list(articles.values)]

    # now get the key authors from the graph

    people = pd.read_sql('''SELECT PMID, LastName, ForeName, S2ID, Au_Order, AuthorNum FROM A02_AuthorList WHERE PMID in {} ORDER BY Au_Order DESC;'''.format(tuple(list(graph.nodes()))), con=conn)


    conn.close()

    citation_dict = {}
    id_to_name = {}
    articles = {} # this is to make sure we only inclue last author
    # import pdb; pdb.set_trace()
    for pmid, lastname, forname, idx, author_order, num_authors in people.values:
        if pmid not in articles:
            articles[pmid] = True
            if idx != 0:
                id_to_name[idx] = forname + ' ' + lastname
                citations = graph.degree(pmid) #removed str(pmid) by guada because it was giving error
                if idx not in citation_dict:
                    citation_dict[idx] = citations
                else:
                    citation_dict[idx] += citations
                    
    citations_ordered = {k: v for k, v in sorted(citation_dict.items(), key=lambda item: item[1], reverse = True)}
    top_k_people = []
    for person in list(citations_ordered.keys())[:k_people]:
        top_k_people.append(id_to_name[person])

    return top_k_papers_pmids, top_k_papers, top_k_people






