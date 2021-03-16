import networkx as nx
import pymysql
import pandas as pd

k_papers = 5 # how many papers to return
k_people = 5 # how many people to return
k_affiliations = 5 # how many affiations to return
host="pubmed-database.c7xgknkzchxj.eu-west-2.rds.amazonaws.com"
port=3306
dbname="pubmed"
user="jks17"
password="password"

def graph_to_recommend(graph, host, port, dbname, user, password):
    pagerank = nx.pagerank(graph)
    pagerank_ordered = {k: v for k, v in sorted(pagerank.items(), key=lambda item: item[1], reverse = True)}
    top_k_papers_pmids = list(pagerank_ordered.keys())[:k_papers + 2]
    # now get the paper titles of these 5 papers and their last author

    conn = pymysql.connect(host=host, user=user,port=port, passwd=password, db = dbname)


    articles = pd.read_sql('''SELECT A01_Articles.PMID, A01_Articles.ArticleTitle, A01_Articles.DateCompleted, 
                A02_AuthorList.LastName, A02_AuthorList.ForeName
                FROM A01_Articles
                JOIN A02_AuthorList ON A02_AuthorList.PMID = A01_Articles.PMID
                WHERE A01_Articles.PMID in {}
                ORDER BY Au_Order;
                '''.format(tuple(list(graph.nodes()))), con=conn)
    
    top_k_papers = []
    papers_to_author = {}
    pmid_to_title = {}
    for pmid, title, year, lastname, forename in articles.values:
        papers_to_author[title + ' ' + str(year)] = forename + ' ' + lastname
        pmid_to_title[str(pmid)] = title + ' ' + str(year)
        if str(pmid) in top_k_papers_pmids:
            if title + ' ' + str(year) not in top_k_papers:
                top_k_papers.append(title + ' ' + str(year))

    # now get the key authors from the graph

    people = pd.read_sql('''SELECT A02_AuthorList.PMID, A02_AuthorList.LastName, A02_AuthorList.ForeName, 
    A02_AuthorList.S2ID, A02_AuthorList.Au_Order, A02_AuthorList.AuthorNum, A13_AffiliationList.Affiliation
        FROM A02_AuthorList
        JOIN A13_AffiliationList ON A13_AffiliationList.PMID = A02_AuthorList.PMID
        WHERE A02_AuthorList.PMID in {} 
        ORDER BY Au_Order DESC;'''.format(tuple(list(graph.nodes()))), con=conn)


    conn.close()


    citation_dict = {} # stores the number of citations for each author
    number_papers_dict = {} # stores the number of papers for each author
    articles = {} 
    authors_to_affiliation = {}
    author_idx_to_name = {}
    affiliation_paper_count = {} # stores the number of papers for each affiliation

    for pmid, lastname, forname, idx, author_order, num_authors, affiliation in people.values:
        authors_to_affiliation[forname + ' ' + lastname] = affiliation
        # need to test to see if we have used this paper before for affiliation information
        if pmid not in articles:
            articles[pmid] = True
            if affiliation not in affiliation_paper_count:
                affiliation_paper_count[affiliation] = 1
            else:
                affiliation_paper_count[affiliation] += 1     
        if idx != 0:
            author_idx_to_name[idx] = forname + ' ' + lastname
            citations = graph.degree(str(pmid))
            if idx not in citation_dict:
                citation_dict[idx] = citations
            else:
                citation_dict[idx] += citations
            if idx not in number_papers_dict:
                number_papers_dict[idx] = 1
            else:
                number_papers_dict[idx] += 1

    affiliation_paper_count_ordered = {k: v for k, v in sorted(affiliation_paper_count.items(), key=lambda item: item[1], reverse = True)}
    # get top k affiliations
    i = 0 
    affiliation_count = {}
    for key in affiliation_paper_count_ordered.keys():
        if i < k_affiliations:
            affiliation_count[key] = affiliation_paper_count_ordered[key]
        i += 1
                
    # now we rank the authors using some kind of metric
    def author_metric(citations, num_papers):
        metric = citations*num_papers
        return metric
    
    metric_dict = {}
    for author in citation_dict.keys():
        metric_dict[author] = author_metric(citation_dict[author], number_papers_dict[author])
                    
    
    metric_dict_ordered = {k: v for k, v in sorted(metric_dict.items(), key=lambda item: item[1], reverse = True)}
    top_k_people = []
    top_k_people_ids = []
    for person in list(metric_dict_ordered.keys())[:k_people]:        
        top_k_people.append(author_idx_to_name[person])
        top_k_people_ids.append(person)

    author_name_to_idx = dict([(value, key) for key, value in author_idx_to_name.items()]) 

    # add people from table 1 to table 2 if they are not in it
    for paper in top_k_papers:
        if papers_to_author[paper] not in top_k_people:
            if papers_to_author[paper] in author_name_to_idx:
                top_k_people.append(papers_to_author[paper])
                top_k_people_ids.append(author_name_to_idx[papers_to_author[paper]])

    return top_k_papers, top_k_papers_pmids, top_k_people, top_k_people_ids, authors_to_affiliation, papers_to_author, citation_dict, number_papers_dict, affiliation_count, pmid_to_title






