
# Pubmed Explorer


## Installation

### Install Manually

```bash
# Create conda environment
conda create --name pubmed python=3.7

# Activate the environment
conda activate pubmed

# Install packages
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
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

## Data

### Official Datasets

The pubmed database is available [here](http://er.tacc.utexas.edu/datasets/ped).

## Usage

### Run dashboard locally

```bash
cd keyword_matching/SKR_Web_API_V2_4/examples
python dash_app.py
```



## Questions

If you have any questions, please create an issue [here](https://github.com/ggonzalezp/pubmedkg_recommender/issues/new).
