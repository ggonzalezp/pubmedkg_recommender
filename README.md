
# Pubmed Explorer

![cover](workflow_figure.png)

## About
The PubMed explorer is designed to be a useful tool for promising students aiming to start a Ph.D. to explore new topics and to find a department and supervisor at the forefront of their field of interest. The tool provides a simple interface, whereby a user can input text or keywords. The input is then used to obtain the corresponding keywords and research field, and the most influential papers in the field as well information regarding 
key authors and departments. 

In addition, a summary for each paper, computed using a Transfomer model [1] operating on the paper's summary, is provided for the user to get an idea of each paper's content at a glance.

Enhanced visualization is provided, optionally, whereby the user can visualize keywords and papers in a 2D space where distances are indicative of paper-paper and paper-keyword similarities. To this aim, we built a heterogeneous graph containing paper and keyword nodes, with paper-paper and paper-keyword edges. We used a multi-layer relational graph convolutional network (RGCN,[2]) architecture to compute paper and keyword embeddings during the optimization of a multi-relational link prediction task on the graph. t-SNE is applied to the embeddings before visualization on the fly. 

## Installation

### Install Manually

```bash
# Create conda environment
conda create --name pubmed python=3.6.9

# Activate the environment
conda activate pubmed

# Install packages
pip install dash==1.19.0
pip install networkx==2.4
pip install pandas==1.0.3
pip install pymysql==1.0.2
pip install simplejson==3.17.2
pip install scikit-learn==0.23.2
pip install transformers==4.4.2


# Compile Java script
cd keyword_matching/SKR_Web_API_V2_4/examples
../compile.sh GenericBatchUser.java
```

Install [pytorch 1.6.0](https://pytorch.org/get-started/previous-versions/) (optionally installing CUDA if you have a CUDA-capable machine)

## Data

### Official Datasets

The pubmed database is available [here](http://er.tacc.utexas.edu/datasets/ped).
Processed files are provided as part of the tool.
Processed files can be downloaded from [here](https://pubmed-explorer-dataset.s3.eu-west-2.amazonaws.com/dict_mesh_pmids.tar.gz), [here](https://pubmed-explorer-dataset.s3.eu-west-2.amazonaws.com/dict_pmid_refs.tar.gz) and [here](https://pubmed-explorer-dataset.s3.eu-west-2.amazonaws.com/out_embeddings.tar.gz). Once downloaded and unzipped, the folder ```out_embeddings``` should be placed in ```graph_embedding```. Folders ```dict_pmid_refs``` and ```dict_mesh_pmids``` should be placed in ```graph_embedding/dataset```


Processed files are: (i) dictionaries with mesh:paper, paper:paper connections (filtered to keep only data from 2015 onwards), and (ii) Paper and keyword embeddings for visualization

In case you wish to process the raw files to generate processed files, the process is as follows:
1. Extract tables to be processed from the MySQL database:

```
cd graph_embedding/dataset
python save_tables.py
```

2. Generate dictionaries:

```
python process_datasets_to_generate_dicts.py
```

3. Generate heterogeneous graph for paper-keyword visualization

```
python process_dataset.py
```

4. Compute data splits for link prediction task:

```
cd ..
python compute_data_link_splits.py
```

4. Evaluate performance of link prediction task on test set:

```
python embedding_het_graph_paper_mesh_rel3.py
```

5. Optimize link prediction task with training set only:

```
python embedding_het_graph_paper_mesh_rel3_only_training.py
```

6.Post-process embeddings:

```
python visualize_embeddings.py
```

Note that to process datasets, you need to install additional dependencies:

```
pip install dask==2021.3.0
```
and install [torch-geometric 1.6.3](https://pytorch-geometric.readthedocs.io/en/1.6.3/notes/installation.html)



## Usage

### Run dashboard locally

```bash
cd keyword_matching/SKR_Web_API_V2_4/examples
python dash_app.py
```



## Questions

If you have any questions, please create an issue [here](https://github.com/ggonzalezp/pubmedkg_recommender/issues/new).

### References 
[1] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, PierricCistac, Tim  Rault, R ́emi Louf, Morgan Funtowicz, Joe  Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing.In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics.

[2] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. Modeling Relational Data with Graph Convolutional Networks. 2017
