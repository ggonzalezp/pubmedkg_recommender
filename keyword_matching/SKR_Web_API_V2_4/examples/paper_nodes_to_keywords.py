import networkx as nx
import pymysql
import pandas as pd

host="pubmed-database.c7xgknkzchxj.eu-west-2.rds.amazonaws.com"
port=3306
dbname="pubmed"
user="jks17"
password="password"

def extract_overlapping_keywords(pmids):
    conn = pymysql.connect(host=host, user=user,port=port,
                           passwd=password, db = dbname)

    print(pmids)

    keywords = pd.read_sql('''SELECT PMID, DescriptorName 
    FROM A06_MeshHeadingList 
    WHERE PMID in {};'''.format(tuple(pmids)), con=conn)

    print(keywords)

    all_keywords = {}
    for pmid, keyword in keywords.values:
        if pmid not in all_keywords:
            all_keywords[pmid] = [keyword]
        else:
            all_keywords[pmid].append(keyword)

    # what keywords appear in all pmids?
    setlist = [set(all_keywords[key]) for key in all_keywords]

    conn.close()
        
    return list(set.intersection(*setlist))