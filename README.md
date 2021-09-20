# <b>DarkVec: Automatic Analysis of Darknet Traffic with Word Embeddings</b>

In this repository we report all artifacts for experiments of the paper _DarkVec: Automatic Analysis of Darknet Traffic with Word Embeddings_
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
    * [`src.utils`](#srcutils)

___
## <b> How to reproduce results in the paper?</b> <a id='howto'></a>

Note: This guide assumes a Linux operating system (tested on Ubuntu 20.04).

1. Clone this repository
2. Download the two gzip files from: https://mplanestore.polito.it:5001/sharing/yDA3EVRFp
3. Unzip the coNEXT.tar.gz file into a subfolder of this repository called `coNEXT`

`gunzip coNEXT.gz`

4. Unzip the raw.tar.gz file into a subfolder of this repository called `raw`

`gunzip raw.gz`

5. Install the required libraries (python3 is assumed):

`pip3 install --user -r requirements.txt`

6. For plotting figures, install the requited fonts (assuming Debian-like Linux):

`sudo apt install cm-super`

7. Run the notebooks described next. For example, to run the first notebook:

`jupyter-lab 01-darknet-overview.ipynb`

Note that the `raw` data is used to create the intermediate datasets in the `coNEXT` folder.
Notebooks are provided (as Appendix) for this step. Given the size of the raw traces a
spark cluster is recommended for this step.

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
* `02-grid-search.ipynb`: experiments performed during the DarkVec grid
search;
* `03-clustering.ipynb`: unsupervised clustering algorithms and results of
the manual clusters inspection;
* `A01-corpus-generation.ipynb`: Appendix1. Code used for generating the
corpus of the experiments. It runs is designed to run on a Spark cluster;
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
## <b>Documentation</b> <a id='doc'></a>

[Back to index](#toc)

In this section we report the documentation for the designed libraries and
utility functions in the `src` folder.

## `src.callbacks` <a id='srccallbacks'></a>

[Back to index](#toc)

Fastplot callback for generating all the figures reported both in the paper and notebooks.

___

```
fig1a(plt, pkts, top)
```


Fastplot callback for generating Fig.1a of the paper.      Port ranking. Zoom on top-14 ports.  Fastplot callback for generating Fig.1a of the paper. Port ranking. Zoom on top-14 ports. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **pkts** *(pandas.DataFrame)*: ECDF of packets per port<br>

- **top** *(pandas.DataFrame)*:

___

```
fig1b(plt, tday)
```



Fastplot callback for generating Fig.1b of the paper.     Sender’s activity pattern.      Fastplot callback for generating Fig.1b of the paper.  Sender’s activity pattern. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **tday** *(pandas.DataFrame)*: timeseries of the ips activeness

___

```
fig2a(plt, cdf)
```



Fastplot callback for generating Fig.2a of the paper.     Amount of packets per sender in 1 month.  Fastplot callback for generating Fig.2a of the paper.  Amount of packets per sender in 1 month. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **cdf** *(pandas.DataFrame)*: packets per senders over a month

___

```
fig2b(plt, cdf, cdf_f)
```



Fastplot callback for generating Fig.2b of the paper.     Cumulative number of senders over time.      Fastplot callback for generating Fig.2b of the paper.  Cumulative number of senders over time. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **cdf** *(pandas.DataFrame)*: cumulative sum of senders over time unfiltered<br>

- **cdf_f** *(pandas.DataFrame)*: cumulative sum of senders over time filtered over 30 days

___

```
fig8a(plt, stretchoid)
```



Fastplot callback for generating Fig.8a of the paper.     Stretchoid activity pattern.  Fastplot callback for generating Fig.8a of the paper.  Stretchoid activity pattern. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **stretchoid** *(pandas.DataFrame)*: sequence of packets per source IP belonging to Stretchoid GT class

___

```
fig8b(plt, en_um)
```



Fastplot callback for generating Fig.8b of the paper.     Engin-Umich activity pattern.  Fastplot callback for generating Fig.8b of the paper.  Engin-Umich activity pattern. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **en_um** *(pandas.DataFrame)*: sequence of packets per source IP belonging to Engin-Umich GT class

___

```
fig5(plt, gs_train_window)
```



Fastplot callback for generating Fig.5 of the paper.     Impact of training window length.  Fastplot callback for generating Fig.5 of the paper.  Impact of training window length. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **gs_train_window** *(pandas.DataFrame)*: results of the experments abount the training window lenght

___

```
fig6(plt, knn_accs)
```



Fastplot callback for generating Fig.6 of the paper.     Impact of k on the k-NN classifier.      Fastplot callback for generating Fig.6 of the paper.  Impact of k on the k-NN classifier. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **knn_accs** *(dict)*: results of the experiments for the impact of classifier k

___

```
fig7a1(plt, heatmaps, Vs, Cs)
```



Fastplot callback for generating the first part of Fig.7a of the paper.     Auto-defined models, grid search through accuracy.      Fastplot callback for generating the first part of Fig.7a of the paper.  Auto-defined models, grid search through accuracy. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **heatmaps** *(list)*: heatmaps resulting from the grid search. Knn classifier accuracy<br>

- **Vs** *(list)*: embedding sizes Vs tested during the grid search<br>

- **Cs** *(list)*: context window sizes Cs tested during the grid search

___

```
fig7a2(plt, heatmaps_time, Vs, Cs)
```



Fastplot callback for generating the second part of Fig.7a of the paper.     Auto-defined models, grid search through model training runtime.  Fastplot callback for generating the second part of Fig.7a of the paper.  Auto-defined models, grid search through model training runtime. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **heatmaps_time** *(list)*: heatmaps resulting from the grid search. Training runtimes<br>

- **Vs** *(list)*: embedding sizes Vs tested during the grid search<br>

- **Cs** *(list)*: context window sizes Cs tested during the grid search

___

```
fig7b1(plt, heatmaps, Vs, Cs)
```



Fastplot callback for generating the first part of Fig.7b of the paper.     Per-service models, grid search through accuracy.      Fastplot callback for generating the first part of Fig.7b of the paper.  Per-service models, grid search through accuracy. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **heatmaps** *(list)*: heatmaps resulting from the grid search. Knn classifier accuracy<br>

- **Vs** *(list)*: embedding sizes Vs tested during the grid search<br>

- **Cs** *(list)*: context window sizes Cs tested during the grid search

___

```
fig7b2(plt, heatmaps_time, Vs, Cs)
```



Fastplot callback for generating the second part of Fig.7b of the paper.     Per-service models, grid search through accuracy.  Fastplot callback for generating the second part of Fig.7b of the paper.  Per-service models, grid search through accuracy. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **heatmaps_time** *(list)*: heatmaps resulting from the grid search. Training runtimes<br>

- **Vs** *(list)*: embedding sizes Vs tested during the grid search<br>

- **Cs** *(list)*: context window sizes Cs tested during the grid search

___

```
fig9(plt, ncs, mods)
```



Fastplot callback for generating Fig.9 of the paper.     Impact of k' in cluster detection.  Fastplot callback for generating Fig.9 of the paper.  Impact of k' in cluster detection. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **ncs** *(list)*: detected number of clusters per tested k'<br>

- **mods** *(list)*: graph modularity with the clusters as partition per tested k'

___

```
fig10(plt, shs)
```



Fastplot callback for generating Fig.10 of the paper.     Average silhouette of points within the found clusters.  Fastplot callback for generating Fig.10 of the paper.  Average silhouette of points within the found clusters. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **shs** *(pandas.DataFrame)*: silhouette plot per cluster

___

```
fig11(plt, clusters, tick)
```



Fastplot callback for generating Fig.11 of the paper.     Activity patterns of Censys sub-clusters.  Fastplot callback for generating Fig.11 of the paper.  Activity patterns of Censys sub-clusters. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **clusters** *(pandas.DataFrame)*: division of the ips in clusters<br>

- **tick** *(pandas.DataFrame)*:

___

```
fig12(plt, clusters)
```



Fastplot callback for generating Fig.12 of the paper.     Activity patterns of Shadowserver sub-clusters.  Fastplot callback for generating Fig.12 of the paper.  Activity patterns of Shadowserver sub-clusters. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **clusters** *(pandas.DataFrame)*: sequence of packets per source IP belonging to the provided clusters

___

```
plot_censys_jaccard(plt, jacc)
```



Fastplot callback for generating the jaccard heatmap. It represents the     jaccard index between the ports contacted by the censys found sub-clusters  Fastplot callback for generating the jaccard heatmap. It represents the  jaccard index between the ports contacted by the censys found sub-clusters <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **jacc** *(pandas.DataFrame)*: jaccard matrix shaped `(n_clusters , n_clusters)`

___

```
plot_generic_pattern(plt, C_)
```



Fastplot callback for plotting the activity patterns of a generic      provided cluster  Fastplot callback for plotting the activity patterns of a generic provided cluster <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **C_** *(pandas.DataFrame)*: activity patterns of IPs partition in clusters to test

___

```
plot_port_pattern(plt, clusters_)
```



Fastplot callback for plotting the pattern of the ports contacted by     IPs belonging to the provided clusters.  Fastplot callback for plotting the pattern of the ports contacted by  IPs belonging to the provided clusters. <br>

#### Parameters<br>

- **plt** *(matplotlib.pyplot)*: matplotlib instance for fastplot callback<br>

- **clusters_** *(pandas.DataFrame)*: set of destination ports timeseries per source IP

## `src.knngraph` <a id='srcknngraph'></a>

[Back to index](#toc)

## KnnGraph

___
```
KnnGraph(graph_path=None,
	graph_gen=False,
	k=4,
	embeddings=None,
    ips=None,
    labels=None)
```

Implementation of the k-Nearest-Neighbors Graph with the Louvain      algorithm application for cluster detection.           The module builds a Graph from a set of embeddings. The nodes are the IP addresses,      and the link among two nodes exists if they belongs to the same      k-neighborhood. The edges weights are the cosine similarity among the nodes     pairs.            The Louvain algorithm is applied and the cluster id of each node is saved      as the attribute `community`.
<br>

#### Parameters<br>

- **graph_path** *(str)*, optional: global path to create a directory named as `model_name` containing the model, scalers and graphs, by default None<br>

- **graph_gen** *(bool)*, optional: if True generate a new knn graph from scratch. If False load an existing .gexf graph, by default False<br>

- **k** *(int)*, optional: number of nearest neighbors to use during the kNN graph creation. IfNone the heuristic for k is performed, by default 4<br>

- **embeddings** *(numpy.ndarray)*, optional: darkvec embeddings used to built the knn graph, by default None<br>

- **ips** *(list)*, optional: set of IPs for which the embeddings must be generated, by default None<br>

- **labels** *(list)*, optional: ground truth labels of the `ips`, by default None. Implementation of the k-Nearest-Neighbors Graph with the Louvain algorithm application for cluster detection.  The module builds a Graph from a set of embeddings. The nodes are the IPs, and the link among two nodes exists if they belongs to the same k-neighborhood. The edges weights are the cosine similarity among the nodes  pairs. The Louvain algorithm is applied and the cluster id of each node is saved as the attribute `community`.   <br>

#### Attributes<br>

- **graph_path** *(str)*: global path to create a directory named as `model_name` containing the model, scalers and graphs.<br>

- **embeddings** *(numpy.ndarray)*: darkvec embeddings used to built the knn graph.<br>

- **labels** *(list)*, optional: ground truth labels of the `ips`.<br>

- **k** *(int)*: number of nearest neighbors to use during the kNN graph creation.<br>

- **gname** *(str)*: `Gknn.gexf` where `k` is the passed value.<br>

- **mod** *(float)*: modularity value of the graph after the Louvain application.<br>

- **comms** *(dict)*: detected communities. The keys are the `ips`, the values are the id of the communities.<br>

- **nc** *(int)*: number of distinct communities.<br>

- **G** *(networkx.classes.graph.Graph))*:

___

```
    get_knn_pos(load_scaler=False, save_scaler=False)
```

#### Parameters<br>

- **load_scaler** *(bool)*, optional: embeddings distance. If False fit a new scaler, by default False<br>

- **save_scaler** *(bool)*, optional: If True save the fitted scaler, by default Falseindices and the relative distances. <br>

#### Returns<br>

- **(tuple)** `(numpy.ndarray, numpy.ndarray)`. Indices of the k nearest neighbors  of each embedding; distances between the embeddings and their k  nearest neighbors.

___

```
    load_graph()
```

<br>

Load an existing .gexf graph.  Load an existing .gexf graph. <br>

#### Returns<br>
- **(networkx.classes.graph.Graph)** k-Nearest-Neighbor Graph built from the generated darkvec embeddings.

___

```
    create_graph(pos, dist)
```

<br>

Take the indices of the k  nearest neighbors of each embeddings and         the distances between the embeddings and their k nearest neighbors. Use         them to create a weighted k-Nearest-Neighbors Graph. The weights are          the cosine similarities among nodes (ips).
<br>

#### Parameters<br>

- **pos** *(numpy.ndarray)*: Indices of the k nearest neighbors of each embedding.<br>

- **dist** *(numpy.ndarray)*: Distances between the embeddings and their k nearest neighbors.Take the indices of the k nearest neighbors of each embeddings and  the distances between the embeddings and their k nearest neighbors. Use  them to create a weighted k-Nearest-Neighbors Graph. The weights are the cosine similarities among nodes (ips). <br>

#### Returns<br>
- **(networkx.classes.graph.Graph)** k-Nearest-Neighbor Graph built from the generated darkvec embeddings.

___

```
    fit_predict(save_graph=False)
```

<br>

Run the Louvain algorithm on the knn graph finding the best nodes          partition and compute the modularity.  Run the Louvain algorithm on the knn graph finding the best nodes partition and compute the modularity. <br>

#### Parameters<br>

- **save_graph** *(bool)*, optional: if True save a .gexf file compatible with Gephi, by default False


## `src.utils` <a id='srcutils'></a>

[Back to index](#toc)



___

```
get_ip_set_by_day(dnet)
```


Get the number of distinc IPs per day
<br>

#### Parameters<br>

- **dnet** *(pandas.DataFrame)*: monthly darknet trafficGet the number of distinc IPs per day <br>

#### Returns<br>
- **(pandas.DataFrame)** distinc IPs per day

___

```
get_ips_ecdf(dnet)
```

<br>

Get the cumulative sum of the distinct IPs per day seen over 30 days of      darknet traffic
<br>

#### Parameters<br>

- **dnet** *(pandas.DataFrame)*: distinc IPs per dayGet the cumulative sum of the distinct IPs per day seen over 30 days of darknet traffic <br>

#### Returns<br>
- **(pandas.DataFrame)** cumulative sum of IPs per day

___

```
get_last_day_stats(df, gt_class)
```

#### Parameters<br>

- **df** *(pandas.DataFrame)*: daily grund truth dataframegt_: strground truth to analyzeGet the number of senders, packets, ports, and the top-5 ports for the provided ground truth   <br>

#### Returns<br>
- **(tuple)** ground truth class, number of senders, number of packets, number of

___

```
load_model(mname)
```

<br>

Load a pre-trained DarkVec model
<br>

#### Parameters<br>

- **mname** *(str)*: name of the modelLoad a pre-trained DarkVec model <br>

#### Returns<br>
- **(gensim.models.word2vec.Word2Vec)** loaded darkvec model

___

```
get_scaled_embeddings(dataset, model, mname, load_scaler = False)
```

<br>

Provide a list of IPs for which the embeddings must be extracted. Then     retrieve the embeddings from the model. Finally scale the embeddings with     a loaded pre-trained scaler or a new one
<br>

#### Parameters<br>

- **dataset** *(pandas.DataFrame)*: source IP and ground truth class<br>

- **model** *(gensim.models.word2vec.Word2Vec)*: darkvec model<br>

- **mname** *(str)*: name of the model<br>

- **load_scaler** *(bool)*, optional: it, by default FalseProvide a list of IPs for which the embeddings must be extracted. Then  retrieve the embeddings from the model. Finally scale the embeddings with  a loaded pre-trained scaler or a new one <br>

#### Returns<br>
- **(pandas.DataFrame)** embeddings indexed by source IP

___

```
split_train_test(data, with_unknown=False)
```

<br>

Prepare the dataset for the Leave-One-Out k-nearest-neighbor classifier.     Fit the classifier with the unkown, then choose if predicting with or      without unknown  <br>


#### Parameters<br>

- **data** *(pandas.DataFrame)*: dataset to split<br>

- **with_unknown** *(bool)*, optional: if True the test dataset has the same shape of the training since the unknown are included. Otherwise, the test dataset has only the knownGT labelled samples, by default FalsePrepare the dataset for the Leave-One-Out k-nearest-neighbor classifier.  Fit the classifier with the unkown, then choose if predicting with or without unknown <br>

#### Returns<br>
- **(tuple)** X train, y train, X test, y test

___

```
get_freqs(x)
```

<br>

Perform the majority voting label assignment on the basis of the k     nearest neighbors
<br>

#### Parameters<br>

- **x** *(numpy.ndarray)*: neighbors labels arrayPerform the majority voting label assignment on the basis of the k  nearest neighbors <br>

#### Returns<br>
- **(str)** majority voting assigned labels

___

```
fit_predict(X_train, y_train, X_test, y_test, k_ = 8)
```

<br>

Run the Leave-One-Out classification. Thus fit the k-nearest-neighbor      classifier and then assign the labels through majority voting
<br>

#### Parameters<br>

- **X_train** *(numpy.ndarray)*: Training embedding dataset shaped `(N_samples,Embedding_size)`<br>

- **y_train** *(numpy.ndarray)*: Training label dataset shaped `(N_samples,)`<br>

- **X_test** *(numpy.ndarray)*: Testing embedding dataset shaped `(N_samples,Embedding_size)` with unlabelled, otherwise `(N_GT_samples,Embedding_size)`<br>

- **y_test** *(numpy.ndarray)*: Testing label dataset shaped `(N_samples,)` with unlabelled, otherwise `(N_GT_samples,)`<br>

- **k_** *(int)*, optional: the `sklearn.neighbors.KNeighborClassifier.kneighbors`, method returnsthe item itself in the first position, by default 8Run the Leave-One-Out classification. Thus fit the k-nearest-neighbor classifier and then assign the labels through majority voting <br>

#### Returns<br>
- **(list)** majority voting assigned labels

___

```
get_shs_df(embeddings, pred)
```

<br>

Compute the silhouette of the provided clusters partition.
<br>

#### Parameters<br>

- **embeddings** *(pandas.DataFrame)*: embeddings dataframe<br>

- **pred** *(numpy.ndarray)*: detected clustersCompute the silhouette of the provided clusters partition. <br>

#### Returns<br>
- **(pandas.DataFrame)** dataset with the clusters and their respective silhouette values

___

```
elbow_eps(distance, nod)
```

<br>

Perform the elbow method on the k-dist plot as described in the DBSCAN     paper
<br>

#### Parameters<br>

- **distance** *(numpy.ndarray)*: distance between samples<br>

- **nod** *(pandas.DataFrame)*: samplesPerform the elbow method on the k-dist plot as described in the DBSCAN  paper <br>

#### Returns<br>
- **(float)** elbow distance point used as epsilon

___

```
extract_cluster(darknet, clusterid)
```

<br>

Extract the cluster traces from the total darknet one
<br>

#### Parameters<br>

- **darknet** *(pandas.DataFrame)*: monthly darknet traces<br>

- **clusterid** *(str)*: identifier of the cluster. Typically it is `Cx`, where x is an integerExtract the cluster traces from the total darknet one <br>

#### Returns<br>
- **(tuple)** monthly cluster traces and heatmap of packets per IP

___

```
Jaccard(x, y)
```

<br>

Compute the jaccard index among the two provided set of ports
<br>

#### Parameters<br>

- **x** *(set)*: set of ports reached by cluster X <br>

- **y** *(set)*: set of ports reached by cluster YCompute the jaccard index among the two provided set of ports <br>

#### Returns<br>
- **(float)** jaccard index

___

```
update_jacc(mat, x, y, sets)
```

<br>

Update an empty jaccard matrix with the provided X and Y clusters
<br>

#### Parameters<br>

- **mat** *(pandas.DataFrame)*: jaccard matrix<br>

- **x** *(str)*: matrix index of cluster X<br>

- **y** *(str)*: matrix index ofcluster Y<br>

- **sets** *(pandas.DataFrame)*: sets of ports reached by different clustersUpdate an empty jaccard matrix with the provided X and Y clusters <br>

#### Returns<br>
- **(pandas.DataFrame)** updated jaccard matrix

___

```
manage_censys_ticks(clusters)
```

<br>

Extract the y-axis ticks centered on the censys sub-cluster scatterplot.
<br>

#### Parameters<br>

- **clusters** *(pandas.DataFrame)*: censys tracesExtract the y-axis ticks centered on the censys sub-cluster scatterplot. <br>

#### Returns<br>
- **(pandas.DataFrame)**

___

```
cluster_report(clusters)
```

<br>

Compute a small report for the provided clusters. Namely, computes     The number of distinct senders, the number of port/protocol pairs and the     top-3 ports
<br>

#### Parameters<br>

- **clusters** *(pandas.DataFrame)*: clusters traces for which obtain the reportCompute a small report for the provided clusters. Namely, computes  The number of distinct senders, the number of port/protocol pairs and the  top-3 ports <br>

#### Returns<br>
- **(pandas.DataFrame)** report of the provided clusters

