# <b>DarkVec: Automatic Analysis of Darknet Traffic with Word Embeddings</b>

## <b>Table Of Content</b> <a id='toc'></a>

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
## <b>Project Structure</b> <a id='proj'></a>

[Back to index](#toc)

### Notebooks <a id='notebook'></a>

[Back to index](#toc)

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

### `src` Folder <a id='src'></a>

[Back to index](#toc)


Python libraries and utilities designed for the experiments: 

* `callbacks.py`: fastplot callbacks for generating the figures of the 
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


All the raw and final data and intermediate preprocessing. They are stored in 
the `DATA` folder of the configuration file. 

### `corpus` Folder <a id='corpus'></a>

[Back to index](#toc)


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

    
### `datasets` Folder <a id='dataset'></a>

[Back to index](#toc)


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

### `gridsearch` Folder <a id='gridsearch'></a>

[Back to index](#toc)


It collects the output of different experiments. They are used to generate the
plots faster than recomputing the results. By running the notebooks it is 
possible to re-create what is in this folder:

* `knngraph.json`: Number of found clusters and modularity during the
testing of k for the knn graph;
* `parameter_tuning.csv`: DarkVec hyperparameters grid search results;
* `training_window.csv`: results of the grid search about the training window;
* `knn_k.csv`: results of the grid search for k f the knn classifier;
* `training_runtime.csv` # TODO

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

#### `services` Folder <a id='services'></a>

[Back to index](#toc)


It reports the `services.json` file. It is a dictionary for the conversion of
the port/protocol pairs of a received packets to a class of services, or 
language.


___
## <b>Documentation</b> <a id='doc'></a>

[Back to index](#toc)


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

- **en_um** *(pandas.DataFrame)*: 

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

KNN graph docstring

## `src.utils` <a id='srcutils'></a>

[Back to index](#toc)

Utils docstring

