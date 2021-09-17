# <b>DarkVec: Automatic Analysis of Darknet Trafficwith Word Embeddings</b>

___
## Project Structure

### Notebooks
The snippets of the experiment discussed in the paper are reported in the 
following notebooks in the main folder: 

* `01-darknet-overview.ipynb`: jupyter notebook performing the darknet 
characterization reported in the paper; 
* `02-grid-search.ipynb`: experiments performed during the DarkVec grid 
search; 
* `03-clustering.ipynb`: unsupervised clustering algorithms and results of
the manual clusters inspection; 
* `A01-corpus-generation.ipynb`: Appendix1. Codes used for generating the
corpus of the experiments. It runs is designed to run on Spark; 
* `A02-model-training.ipynb`: Appendix2. Training of the models used in the
paper; 
* `A03-darknet-interim`: Appendix3. Some intermediate preprocessing. To 
reduce the notebooks runtime, we save intermediate dataframes and load them
instead of recomputing them. In this notebook, user can observe and repeat
such preprocessing. 

### `src` Folder
Python libraries and utilities designed for the experiments: 

* `callbacks.py`: fastplot callbacks for generating the figures of the 
paper; 
* `knngraph.py`: implementation of the k-nearest-neighbor-graph described
in the paper; 
* `utils.py`: some utility functions; 

### Configuration file `config.py`
For running the experiments, it could be necessary to change the global paths
managing the files. In this case, it should be sufficient to replace the 
following: 


    # global path of the raw traces
    TRACES

    # global path of the data folder
    DATA 


for the other parameters see the `config.py` file.


___
## Data Description
All the raw and final data and intermediate preprocessing. They are stored in 
the `DATA` folder of the configuration file. 

### `corpus` Folder 
It contains all the corpora generated for the experiments. A part from the 
IP2VEC case, a corpus is a set of .txt files reporting a sequence of IPs wrt.
different languages, or classes of service. 

The corpora we provide are: 

* `dante5`. Last 5 days of collected traffic used in the DANTE paper; 
* `darkvec30auto`. Last 30 days of collected traffic used in the DarkVec 
experiments. Auto-defined languages; 
* `darkvec30single`. Last 30 days of collected traffic used in the DarkVec 
experiments. Single language; 
* `darkvec30xserv`. Last 30 days of collected traffic used in the DarkVec 
experiments. Per-service languages; 
* `darkvec5xserv`. Last 5 days of collected traffic used in the DarkVec 
experiments. Per-service languages; 
* `ip2vec5`. Last 5 days of collected traffic used in the IP2VEC paper. 

    
### `datasets` Folder
It contains all the intermediate preprocessing. Namely: 

* `darknet.csv.gz`: full 30 days of unfiltered darknet traffic; 
* `darknet_d5.csv.gz`: last 5 days of unfiltered darknet traffic;
* `embeddings_ip2vec.csv.gz`: embeddings generated thrugh the IP2VEC 
methodology after 5 days of training;
* `darknet_d1.csv.gz`: last day of darknet traffic unfiltered;
* `detected_clusters.csv.gz`: results of the GMA on the knn graph of the 
paper;
* `darknet_d1_f5.csv.gz`: last day of darknet traffic filtered over the 
last 5 days;
* `embeddings_d1_f30.csv.gz`: last day of darknet traffic filtered over the 
last 30 days;
* `sh_cluster.csv.gz`: per-cluster silhuette dataset;
*`ips.json`: it contains all the list of IPs referred to a certain day. 
Each key indicates a day:
    - `d30_u`: it is referred to the 30 days dataset unfiltered;
    - `d30_f`: 30 days dataset filtered;
    - `d1_u`: last day unfiltered;
    - `d1_f30`: last day filtered over 30 days;

### `gridsearch` Folder 
It collects the output of different experiments. They are used to generate the
plots faster than recomputing the results. By running the notebooks it is 
possible to re-create what is in this folder:

* `knngraph.json`: Number of found clusters and modularity during the
testing of k for the knn graph;
* `parameter_tuning.csv`: DarkVec hyperparameters grid search results;
* `training_window.csv`: results of the grid search about the training window;
* `knn_k.csv`: results of the grid search for k f the knn classifier;
* `training_runtime.csv` # TODO

### `groundtruth` Folder
It contains the gorund truth we generated in `lsground_truth_full.csv.gz`. It 
is a collection of IP addresses with the respective label. The label may be 
like `sonar` if the IP belongs to the Project Sonar, `shodan` if it belongs to 
the Shodan pool.  


### `models` Folder
Model trained during the experiments of the paper. The model names are 
related to the parameters `C`, or context window size, and `V` or embeddings
size. Namely, they are:

* `single_cC_vV_iter20`: DarkVec model trained on 30 days. Single language;
* `auto_cC_vV_iter20`:  DarkVec model trained on 30 days. Auto-defined 
languages;
* `service_cC_vV_iter20`: DarkVec model trained on 30 days. Per-service 
languages;
* `fivedays_c25_v50_iter20`: DarkVec model trained on 5 days. Per-service 
languages;
* `ip2vec5embedder`: keras embedder generated through our implementation 
following the IP2VEC paper.

#### `services` Folder
It reports the `services.json` file. It is a dictionary for the conversion of
the port/protocol pairs of a received packets to a class of services, or 
language.


___
## Documentation

## `src.callbacks`
Callbacks docstring

## `src.knngraph`
KNN graph docstring

## `src.utils`
Utils docstring

