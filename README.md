# <b>DarkVec: Automatic Analysis of Darknet Trafficwith Word Embeddings</b>

___
## Project Structure
* Notebooks:
    The snippets of the experiment discussed in the paper are reported in the following notebooks in the main folder:
    * `config.py`: description of configuration file
    * `01-darknet-overview.ipynb`: description of the first notebook
    * `02-grid-search.ipynb`: description of he grid search notebook
    * `03-clustering.ipynb`: description of the clustering notebook
    * `A01-corpus-generation.ipynb`: codes used during the corpus generation for the experiments. It runs is designed to run on Spark
* `src` folder:
    * `callbacks.py`
    * `knngraph.py`
    * `utils.py`
* `data` folder:
    * `clusterinspection` folder
    * `corpus` folder
    * `dataset` folder
    * `gridsearch` folder
    * `ip2vec` folder
    * `models` folder
   

-`ips.json`: it contains all the list of IPs referred to a certain day. Each key indicates a day:
    - `d30_u`: it is referred to the 30 days dataset unfiltered
    - `d30_f`: 30 days dataset filtered
    - `d1_u`: last day unfiltered
    - `d1_f30`: last day filtered over 30 days
   
___
## Documentation

## `src.callbacks`
Callbacks docstring

## `src.knngraph`
KNN graph docstring

## `src.utils`
Utils docstring

___
## ToDo
Update config