import matplotlib.pyplot as plt
import pandas as pd
import json
from config import *
from src.utils import split_train_test, fit_predict

def convert_pp(x, services):       
    """Convert the port/protocol pair of a packet in the respective service

    Parameters
    ----------
    x : str
        port/protocol pair
    services : dict
        domain knowledge based class of service

    Returns
    -------
    str
        domain knowledge based class of service the packet belongs to
    """
    try: x1 = services[x]
    except: x1 = unknown_class(x)
    
    return x1

def unknown_class(x):
    """Manage the port/protocol pairs that are not classified in `services`

    Parameters
    ----------
    x : str
        port/protocol pair

    Returns
    -------
    str
        conversion
    """
    x = x.split('/')[0]
    #System Ports
    if x!='-':
        if int(x) >= 0 and int(x) <= 1023: return 'unk_sys'
        # User Ports
        elif int(x) >= 1024 and int(x) <= 49151:return 'unk_usr'
        # Ephemeral Ports
        elif int(x) >= 49152 and int(x) <= 65535:return 'unk_eph'
    else:
        return 'icmp'
    

def extract_features(baseline_df, ktop):
    """Extract the features for the baseline (top-`ktop` ports of each GT class)

    Parameters
    ----------
    baseline_df : pandas.DataFrame
        raw traces of the baseline
    ktop : int
        number of GT top ports to consider as feature

    Returns
    -------
    numpy.ndarray
        list of port/protocol pairs used as features
    """
    # Get the top-ports by classes
    temp = baseline_df.copy()
    rows = []
    # Get the number of packets per port.
    temp = temp.groupby(['class', 'pp']).agg({'pkts':sum}).reset_index()
    # For each class
    for c in temp['class'].unique():
        # Isolate the class
        temp_ = temp[temp['class'] == c]
        # Sort by number of packets
        top = temp_.sort_values('pkts', ascending=False)
        # Get the top-k ports, and packets of the class
        if ktop == 1:
            top_port = top.pp.values[:1]
            top_pkts = top.pkts.values[:1]
        else:
            top_port = top.pp.values[:ktop-1]
            top_pkts = top.pkts.values[:ktop-1]
        for i in range(top_port.shape[0]):
            rows.append((c, top_port[i], top_pkts[i]))
    features = pd.DataFrame(rows)[1].unique()

    return features

def pivot_baseline(baseline_df, features):
    """Extract the dataset from the raw one for the dataset. The features are
    the percentage of traffic sent by each IP to the port/protocol pairs listed
    in `features`

    Parameters
    ----------
    baseline_df : pandas.DataFrame
        raw baseline traces
    features : numpy.ndarray
        list of port/protocol pairs used as features

    Returns
    -------
    pandas.DataFrame
        final baseline dataset
    """
    # Build the dataset as the fraction of daily packets per top-k*9 ports
    pivot = baseline_df.pivot_table(values='pkts', index='ip', 
                                    columns='pp', aggfunc='sum')\
              .reindex(columns=features).fillna(.0)
    dataset = pivot.to_numpy()/baseline_df.groupby('ip')\
                                          .agg({'pkts':'sum'})\
                                          .to_numpy()
    dataset = pd.DataFrame(dataset, columns=pivot.columns, index=pivot.index)
    dataset = dataset.reset_index()\
                     .merge(baseline_df[['ip', 'class']].drop_duplicates(), 
                            on='ip')\
                     .set_index('ip')
    
    return dataset

def build_dataset_from_raw(raw_df, top_k_ports):
    """Run the full baseline pipeline starting from raw data, extracting 
    features and generating the final dataset

    Parameters
    ----------
    raw_df : pandas.DataFrame
        raw baseline traces
    top_k_ports : int
        number of ground truth top ports to consider as features

    Returns
    -------
    pandas.DataFrame
        final baseline dataset
    """
    features = extract_features(raw_df, top_k_ports)
    dataset = pivot_baseline(raw_df, features)
    
    return dataset


def knn_simple_step(dataset, with_unknown, k):
    """Run a k-nearest-neighbor fit and predict

    Parameters
    ----------
    dataset : pandas.DataFrame
        dataset to classify
    with_unknown : bool
        if True the predicting dataset is the same of the fitting, otherwise
        the unknown labelled IPs are not classified
    k : int
        number of nearest neughbors to consider in the majority voting label
        assignment

    Returns
    -------
    tuple
        list of y true and y predicted labels
    """
    # Run the kNN classifier
    X_train, y_train, X_test, y_test = split_train_test(dataset, 
                                            with_unknown=with_unknown)
    y_true = y_test
    y_pred = fit_predict(X_train, y_train, X_test, y_test, k_ = k+1)
    
    return y_true, y_pred

def pivot_clusters(dataset):
    """Generate the dataframe after the supervised k-means for the heatmap

    Parameters
    ----------
    dataset : pandas.DataFrame
        dataset to process

    Returns
    -------
    pandas.DataFrame
        (`N_GT_class x N_clusters`) shaped dataset. In can be visualized as a
        heatmap
    """
    temp = dataset.copy()
    temp['cnt'] = 1
    pivot = temp.pivot_table(values = 'cnt', index='class', 
                             columns='cluster', aggfunc='sum')
    pivot = pivot.divide(pivot.sum(1), 'rows')*100

    return pivot