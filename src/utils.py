import pandas as pd
import pickle
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples
import kneed 
import numpy as np
from config import *


#=============================================================================#
# FIRST NOTEBOOK 01-darknet-overview
# Utility functions used mainly in the codes of the first notebook
#=============================================================================#

def get_ip_set_by_day(dnet):
    """Get the number of distinc IPs per day

    Parameters
    ----------
    dnet : pandas.DataFrame
        monthly darknet traffic

    Returns
    -------
    pandas.DataFrame
        distinc IPs per day
    """
    dnet.index = pd.DatetimeIndex(dnet.ts)
    dnet = dnet.resample('D').agg({'ip':set})
    dnet.loc['2021-03-02', 'ip'] = dnet.iloc[1]['ip'].union(dnet.iloc[0]['ip'])
    dnet = dnet.iloc[1:]
    
    return dnet

def get_ips_ecdf(dnet):
    """Get the cumulative sum of the distinct IPs per day seen over 30 days of 
    darknet traffic

    Parameters
    ----------
    dnet : pandas.DataFrame
        distinc IPs per day

    Returns
    -------
    pandas.DataFrame
        cumulative sum of IPs per day
    """
    prev = dnet.iloc[0]['ip']
    day = dnet.index[0]
    cumuls = [(day, len(prev))]
    for i in range(1, dnet.shape[0]):
        day = dnet.index[i]
        new = dnet.iloc[i]['ip'].union(prev)
        cumuls.append((day, len(new)))
        prev = new
        
    return cumuls

def get_last_day_stats(df, gt_class):
    """Get the number of senders, packets, ports, and the top-5 ports for the 
    provided ground truth class 

    Parameters
    ----------
    df : pandas.DataFrame
        daily grund truth dataframe
    gt_class : str
        ground truth class to analyze

    Returns
    -------
    tuple
        ground truth class, number of senders, number of packets, number of 
        ports and top-5 ports
    """
    senders   = df[df['class'] == gt_class].ip.unique().shape[0]
    ports     = df[df['class'] == gt_class].pp.unique().shape[0]
    packets   = df[df['class'] == gt_class].shape[0]
    top5ports = df[df['class'] == gt_class].value_counts('pp')[:5]
    
    top5 = ''
    for top_p in top5ports.index:
        port_traffic = df[(df['class'] == gt_class) & (df.pp == top_p)]
        port_traffic_perc = port_traffic.shape[0]*100/packets
        top5+=f'{top_p}({round(port_traffic_perc, 1)}%), '
    top5 = top5[:-2]
    
    return (gt_class, senders, packets, ports, top5)


#=============================================================================#
# SECOND NOTEBOOK 02-grid-search
# Utility functions used mainly in the codes of the second notebook
#=============================================================================#

def load_model(mname):
    """Load a pre-trained DarkVec model

    Parameters
    ----------
    mname : str
        name of the model

    Returns
    -------
    gensim.models.word2vec.Word2Vec
        loaded darkvec model
    """
    model_path_name = f'{MODELS}/{mname}.model'
    model = Word2Vec.load(model_path_name, mmap='r')
        
    return model

def get_scaled_embeddings(dataset, model, mname, load_scaler = False):
    """Provide a list of IPs for which the embeddings must be extracted. Then
    retrieve the embeddings from the model. Finally scale the embeddings with
    a loaded pre-trained scaler or a new one

    Parameters
    ----------
    dataset : pandas.DataFrame
        source IP and ground truth class
    model : gensim.models.word2vec.Word2Vec
        darkvec model
    mname : str
        name of the model
    load_scaler : bool, optional
        if True load a pre-generated scaler, otherwise fit a new one and save 
        it, by default False

    Returns
    -------
    pandas.DataFrame
        embeddings indexed by source IP
    """
    embeddings = {}
    failed = []
    for x in dataset.ip.unique():
        try: embeddings[x] = model.wv.__getitem__(x)
        except: failed.append(x)
    
    embs = pd.DataFrame(embeddings).T
    embs = embs.reset_index().rename(columns={'index':'ip'})\
               .merge(dataset, on='ip', how='left').set_index('ip')

    # Scale
    scaler_path_name = f'{MODELS}/{mname}.scaler'
    if not load_scaler:
        scaler = StandardScaler().fit(embs[embs.columns[:-1]])
        with open(scaler_path_name, 'wb') as file: pickle.dump(scaler, file)      
    else:
        with open(scaler_path_name, 'rb') as file: scaler = pickle.load(file)
    
    scaled = scaler.transform(embs[embs.columns[:-1]])
    embs[embs.columns[:-1]] = scaled
      
    return embs

def split_train_test(data, with_unknown=False):
    """Prepare the dataset for the Leave-One-Out k-nearest-neighbor classifier.
    Fit the classifier with the unkown, then choose if predicting with or 
    without unknown

    Parameters
    ----------
    data : pandas.DataFrame
        dataset to split
    with_unknown : bool, optional
        if True the test dataset has the same shape of the training since the 
        unknown are included. Otherwise, the test dataset has only the known
        GT class labelled samples, by default False

    Returns
    -------
    tuple
        X train, y train, X test, y test
    """
    X_train = data[data.columns[:-1]].values
    y_train = data['class'].values
    if not with_unknown:
        X_test = data[(data['class']!='unknown')][data.columns[:-1]].values
        y_test = data[(data['class']!='unknown')]['class'].values
    else:
        X_test = data[data.columns[:-1]].values
        y_test = data['class'].values
    
    return X_train, y_train, X_test, y_test

def get_freqs(x):
    """Perform the majority voting label assignment on the basis of the k
    nearest neighbors

    Parameters
    ----------
    x : numpy.ndarray
        neighbors labels array

    Returns
    -------
    str
        majority voting assigned labels
    """
    lab, freqs = np.unique(x, return_counts=True)
    if lab.shape[0] == 1:
        return np.array(lab[0])
    else:
        if freqs[0]==freqs[1]:
            return np.array('n.a.')
        else:
            return np.array(lab[0])
        
def fit_predict(X_train, y_train, X_test, y_test, k_ = 8):
    """Run the Leave-One-Out classification. Thus fit the k-nearest-neighbor 
    classifier and then assign the labels through majority voting

    Parameters
    ----------
    X_train : numpy.ndarray
        Training embedding dataset shaped `(N_samples,Embedding_size)`
    y_train : numpy.ndarray
        Training label dataset shaped `(N_samples,)`
    X_test : numpy.ndarray
        Testing embedding dataset shaped `(N_samples,Embedding_size)` with 
        unlabelled, otherwise `(N_GT_samples,Embedding_size)`
    y_test : numpy.ndarray
        Testing label dataset shaped `(N_samples,)` with unlabelled, otherwise 
        `(N_GT_samples,)`
    k_ : int, optional
        k of the knn classifier. The actual k is k-1. The plus one is because
        the `sklearn.neighbors.KNeighborClassifier.kneighbors`, method returns
        the item itself in the first position, by default 8

    Returns
    -------
    list
        majority voting assigned labels
    """
    knn = knn = KNN(n_neighbors=k_, metric='cosine', n_jobs=-1)
    knn.fit(X_train, y_train)
    pred_idx = knn.kneighbors(X_test)[1][:, 1:]
    pred_lab = y_train[pred_idx]
    y_pred = [get_freqs(i) for i in pred_lab]
    
    return y_pred


#=============================================================================#
# THIRD NOTEBOOK 03-clustering
# Utility functions used mainly in the codes of the third notebook
#=============================================================================#

def get_shs_df(embeddings, pred):
    """Compute the silhouette of the provided clusters partition.

    Parameters
    ----------
    embeddings : pandas.DataFrame
        embeddings dataframe
    pred : numpy.ndarray
        detected clusters

    Returns
    -------
    pandas.DataFrame
        dataset with the clusters and their respective silhouette values
    """
    embeddings_df = pd.DataFrame(pred, columns=['C'], index=embeddings.index)
    silhouettes = silhouette_samples(X=embeddings[embeddings.columns[:-1]],
                                     labels = embeddings_df.C.values, 
                                     metric='cosine')

    sh_df = pd.DataFrame(silhouettes, columns=['sh'], 
                         index=embeddings_df.index)
    sh_df['C'] = embeddings_df.C.values
    
    return sh_df

def elbow_eps(distance, nod):
    """Perform the elbow method on the k-dist plot as described in the DBSCAN
    paper

    Parameters
    ----------
    distance : numpy.ndarray
        distance between samples
    nod : pandas.DataFrame
        samples

    Returns
    -------
    float
        elbow distance point used as epsilon
    """
    k = int(np.log(nod.shape[0])-1)
    nn = NearestNeighbors(n_neighbors = k, metric='precomputed').fit(distance)
    ds, idx = nn.kneighbors()
    srtd = np.sort(-ds[:,-1])
    srtd = -srtd/nod.shape[1]
    kneedle = kneed.KneeLocator(
        range(srtd.shape[0]), srtd, curve="convex", 
        direction="decreasing", S=10
    )
    return srtd[kneedle.knee]

def extract_cluster(darknet, clusterid):
    """Extract the cluster traces from the total darknet one

    Parameters
    ----------
    darknet : pandas.DataFrame
        monthly darknet traces
    clusterid : str
        identifier of the cluster. Typically it is `Cx`, where x is an integer

    Returns
    -------
    tuple
        monthly cluster traces and heatmap of packets per IP
    """
    clusterX = darknet[darknet.C == clusterid]
    CX = clusterX.pivot_table(index = 'ip', columns = 'pp', values='pkts', 
                              aggfunc='sum')
    clusterX['class'].unique()
    
    return clusterX, CX

def Jaccard(x, y):
    """Compute the jaccard index among the two provided set of ports

    Parameters
    ----------
    x : set
        set of ports reached by cluster X 
    y : set
        set of ports reached by cluster Y

    Returns
    -------
    float
        jaccard index
    """
    inter = x.intersection(y)
    union = x.union(y)
    
    return len(inter)/len(union)

def update_jacc(mat, x, y, sets):
    """Update an empty jaccard matrix with the provided X and Y clusters

    Parameters
    ----------
    mat : pandas.DataFrame
        jaccard matrix
    x : str
        matrix index of cluster X
    y : str
        matrix index of  cluster Y
    sets : pandas.DataFrame
        sets of ports reached by different clusters

    Returns
    -------
    pandas.DataFrame
        updated jaccard matrix
    """
    c1_ = sets.loc[x].pp
    c2_ = sets.loc[y].pp
    mat.loc[x, y] = Jaccard(c1_, c2_)
    
    return mat

def manage_censys_ticks(clusters):
    """Extract the y-axis ticks centered on the censys sub-cluster scatterplot.

    Parameters
    ----------
    clusters : pandas.DataFrame
        censys traces

    Returns
    -------
    pandas.DataFrame
        assigned ticks for the y-axis
    """
    tick = clusters.groupby('C').agg({'ip':lambda x: len(set(x))}).sort_index()
    tick['cv'] = tick.ip.apply(lambda x: int(x/2))
    tick['x'] = 1
    for i in range(tick.shape[0]):
        idx = tick.index[i]
        if i == 0:
            tick.loc[idx, 'x'] = tick.loc[idx, 'cv']
        else:
            idx_prev = tick.index[i-1]
            tick.loc[idx, 'x'] = tick.loc[idx_prev,'x']+tick.loc[idx_prev,'ip']\
                                -tick.loc[idx_prev,'cv']+tick.loc[idx,'cv']
    
    return tick

def cluster_report(clusters):
    """Compute a small report for the provided clusters. Namely, computes
    The number of distinct senders, the number of port/protocol pairs and the
    top-3 ports

    Parameters
    ----------
    clusters : pandas.DataFrame
        clusters traces for which obtain the report

    Returns
    -------
    pandas.DataFrame
        report of the provided clusters
    """
    # Top ports
    txts = []
    for c in clusters.C.unique():
        temp = clusters[clusters.C==c]
        top3 = (temp.value_counts('pp')*100/temp.shape[0])[:3]
        txt = '  '.join([f'{p}({round(top3[p], 1)})%' for p in top3.index])
        txts.append((c, txt))

    # General
    rep = clusters.groupby('C').agg({'ip':lambda x: len(set(x)),
                                     'pp':lambda x: len(set(x))})\
                               .rename(columns={'ip':'Senders', 
                                                'pp': 'Port/Protocol'})\
                               .reset_index().merge(
                                   pd.DataFrame(txts, 
                                                columns=['C', 'Top3 Ports']))
    
    return rep