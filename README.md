# <b>DarkVec: Automatic Analysis of Darknet Traffic with Word Embeddings</b>

In this repository we report all artifacts for experiments of the paper _DarkVec: Automatic Analysis of Darknet Traffic with Word Embeddings_. The current version is _v2_ after the paper review. in the [changelog](#changelog) session the main changes are reported.
___
***Note:*** All source code and data we provide are the ones included in the paper. We provide the source code and a description for generating the intermediate preprocessing files with the obtained results. To speed up the notebook execution, by default we trim the file loading.

Please, note that when running the code without starting from the provided intermediate files, or because of random seeds used in third-party libraries, some results may slightly chage from one run to another. The general trends observed in the paper are however stable.

Notice that this repository has already been updated to include novel experiments and some changes requested by the reviewers. More changes will be included in the coming weeks to reflect the camera-ready version.

## <b>Table Of Content</b> <a id='toc'></a>

* [How to reproduce results in the paper?](#howto)
* [Project Structure](#proj)
    * [Notebooks](#notebook)
    * [`src` Folder](#src)
    * [Configuration File `config.py`](#config)
* [Data Description](#data)
    * [`corpus` Folder](#corpus)
    * [`datasets` Folder](#dataset)
    * [`gridsearch` Folder](#gridsearch)
    * [`groundtruth` Folder](#groundtruth)
    * [`models` Folder](#models)
    * [`services` Folder](#services)
* [Documentation](#doc)
    * [`src.callbacks`](#srccallbacks)
    * [`src.knngraph`](#srcknngraph)
    * [`src.kmeans`](#srckmeans) 
    * [`src.utils`](#srcutils)
    * [`src.review`](#srcreview)
* [Changelog](#changelog)

___
## <b> How to reproduce results in the paper?</b> <a id='howto'></a>

Note: This guide assumes a Debian-like system (tested on Ubuntu 20.04 & Debian 11).

1. Clone this repository
2. Download the two gzip files from: https://mplanestore.polito.it:5001/sharing/3cBHvHD5h

A password is required to download the traces above. Contact the authors in case you 
want to have access to the data. 
The reason for this protection is that some IP addresses sending traffic
to the darknet can be victims of attacks, such as people that have the PC hacked
and take part on scan activity without their knowledge.


3. Unzip the coNEXT.tar.gz file into a subfolder of this repository called `coNEXT`

`tar -zxvf coNEXT.tar.gz`

4. Unzip the raw.tar.gz file into a subfolder of this repository called `raw`

`tar -zxvf raw.tar.gz`

5. Install the `virtualenv` library (python3 is assumed):

`pip3 install --user virtualenv`

6. Create a new virtual environment and activate it:

```
virtualenv darkvec-env
source darkvec-env/bin/activate
```

7. Install the required libraries (python3 is assumed):

`pip3 install -r requirements.txt`

<!--
8. For plotting figures, install the required fonts (assuming Debian-like Linux):

`sudo apt install dvipng texlive-latex-extra texlive-fonts-recommended cm-super`

9. Run the notebooks described next. For example, to run the first notebook:

`jupyter-lab 01-darknet-overview.ipynb`
-->
8. Run the notebooks described next. For example, to run the first notebook:

`jupyter-lab 01-darknet-overview.ipynb`

Note that the `raw` data is used to create the intermediate datasets in the `coNEXT` folder.
Notebooks are provided (as Appendix) for this step. Given the size of the raw traces a
spark cluster is recommended for this step.

10. When the notebook exploration is ended, remember to deactivate the virtual environment:

`deactivate`


[Back to index](#toc)

## <b>Project Structure</b> <a id='proj'></a>

[Back to index](#toc)

Firstly we provide an overview of the folders and data in the project.

### Notebooks <a id='notebook'></a>

[Back to index](#toc)

The experiments discussed in the paper are reported in the
following notebooks in the main folder:

* `01-darknet-overview.ipynb`: jupyter notebook performing the darknet
characterization reported in the paper;
* `02-baseline.ipynb`: experiments performed with the supervised approach. 
Section Baseline of the paper;
* `03-grid-search.ipynb`: unsupervised clustering algorithms and results of
the manual clusters inspection;
* `04-clustering.ipynb`: experiments performed during the DarkVec grid
search.

The previous notebooks start from intermediate datasets and pre-trained models.
These steps are time consuming, in particular for alternative approaches.
They can be reproduced with the following notebooks:

* `A01-corpus-generation.ipynb`: Appendix1. Code used for generating the
corpus of the experiments. Ideally, it should be run on a Spark cluster.
The provided notebook is steup for spark stand-alone, which is not scalable;
* `A02-model-training.ipynb`: Appendix2. Training of the models used in the
paper (requires Gensim);
* `A03-darknet-interim`: Appendix3. Some intermediate preprocessing. To
reduce the notebook runtime, we save intermediate dataframes and load them
instead of always recomputing everything. In this notebook, user can observe and repeat
such preprocessing.

### `src` Folder <a id='src'></a>

[Back to index](#toc)


Python libraries and utilities designed for the experiments:

* `callbacks.py`: ``fastplot'' callbacks for generating the figures of the
paper;
* `knngraph.py`: implementation of the k-nearest-neighbor-graph described
in the paper;
* `utils.py`: some utility functions;
* `review.py`: utility functions used in notebooks after the paper review. 
Mainly used in the baseline experiments;
* `kmeans.py`: implementation of the supervised k-Means algorithm described
in the paper;

### Configuration File `config.py` <a id='config'></a>

[Back to index](#toc)


For running the experiments, it could be necessary to change the global paths
managing the files. In this case, it should be sufficient to replace the
following:


    # global path of the raw traces
    TRACES

    # global path of the data folder
    DATA


for the other parameters see the `config.py` file.


___
## <b>Data Description</b> <a id='data'></a>

[Back to index](#toc)


All raw and intermediate data resulting from preprocessing steps can be found in
the `DATA` folder, pointed by the configuration file.

### `corpus` Folder <a id='corpus'></a>

[Back to index](#toc)


It contains all the corpora generated for the experiments. Except for the
IP2VEC case, a corpus is a set of .txt files reporting a sequence of IPs wrt.
different languages (classes of service).

The corpora we provide are:

* `dante5`. Last 5 days of collected traffic used as in the DANTE paper;
* `darkvec30auto`. Last 30 days of collected traffic used in the DarkVec
experiments. Auto-defined languages;
* `darkvec30single`. Last 30 days of collected traffic used in the DarkVec
experiments. Single language;
* `darkvec30xserv`. Last 30 days of collected traffic used in the DarkVec
experiments. Per-service languages;
* `darkvec5xserv`. Last 5 days of collected traffic used in the DarkVec
experiments. Per-service languages;
* `ip2vec5`. Last 5 days of collected traffic used in the IP2VEC paper.


### `datasets` Folder <a id='dataset'></a>

[Back to index](#toc)


It contains all the intermediate preprocessing. Namely:

* `darknet.csv.gz`: full 30 days of unfiltered darknet traffic;
* `darknet_d5.csv.gz`: last 5 days of unfiltered darknet traffic;
* `embeddings_ip2vec.csv.gz`: embeddings generated thrugh the IP2VEC
methodology after 5 days of training;
* `darknet_d1.csv.gz`: last day of darknet traffic unfiltered;
* `detected_clusters.csv.gz`: results of the Greedy Modularity Algorithm (aka Louvain)
on the knn graph of the paper;
* `darknet_d1_f5.csv.gz`: last day of darknet traffic filtered over the
last 5 days;
* `embeddings_d1_f30.csv.gz`: last day of darknet traffic filtered over the
last 30 days;
* `sh_cluster.csv.gz`: per-cluster silhuette dataset;
*`ips.json`: it contains all the list of IP addresses referred to a certain day.
Each key indicates a day:
    - `d30_u`: it is referred to the 30 days dataset unfiltered;
    - `d30_f`: 30 days dataset filtered;
    - `d1_u`: last day unfiltered;
    - `d1_f30`: last day filtered over 30 days;

### `gridsearch` Folder <a id='gridsearch'></a>

[Back to index](#toc)

It collects the output of different experiments. They are used to generate the
plots without recomputing the results. By running the notebooks it is
possible to re-create what is in this folder:

* `knngraph.json`: Number of found clusters and modularity during the
testing of k for the knn graph;
* `parameter_tuning.csv`: DarkVec hyperparameters grid search results;
* `training_window.csv`: results of the grid search about the training window;
* `knn_k.csv`: results of the grid search for k f the knn classifier;

### `groundtruth` Folder <a id='groundtruth'></a>

[Back to index](#toc)


It contains the gorund truth we generated in `lsground_truth_full.csv.gz`. It
is a collection of IP addresses with the respective label. The label may be
like `sonar` if the IP belongs to the Project Sonar, `shodan` if it belongs to
the Shodan pool.


### `models` Folder <a id='models'></a>

[Back to index](#toc)


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

### `services` Folder <a id='services'></a>

[Back to index](#toc)


It reports the `services.json` file. It is a dictionary for the conversion of
the port/protocol pairs of a received packets to a class of services, or
language.


___
