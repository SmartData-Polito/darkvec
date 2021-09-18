"""Fastplot callback for generating all the figures reported both in the paper
and notebooks.
"""

import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
from datetime import datetime


def fig1a(plt, pkts, top):
    """Fastplot callback for generating Fig.1a of the paper. 
    Port ranking. Zoom on top-14 ports.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    pkts : pandas.DataFrame
        ECDF of packets per port
    top : pandas.DataFrame
        ECDF of packets per top-14 ports
    """
    plt.plot(pkts.pkts.values, linewidth=1)
    plt.grid(linestyle='--')
    plt.xlabel('Port rank')
    plt.ylabel('ECDF')
    plt.ylim(0, 1.1)
    plt.xlim(-1000, 66000)
    # Top-14 ports zoom
    a = plt.axes([.38, .33, .55, .4])
    plt.plot(pkts.pkts.values[:14])
    plt.xticks(range(14), top.port, rotation=50)
    plt.xlim(-.5, 13.5)
    plt.grid(linestyle='--')
    
def fig1b(plt, tday):
    """Fastplot callback for generating Fig.1b of the paper.
    Sender’s activity pattern.
    
    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    tday : pandas.DataFrame
        timeseries of the ips activeness
    """
    plt.scatter(tday.index, tday.tkn, s=.000005, marker='o')
    plt.grid(linestyle='--')
    plt.xlabel('Time [day]')
    plt.ylabel('Sender')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlim(datetime.strptime('2021-03-03:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-03-12:01', '%Y-%m-%d:%H'))   
    plt.ylim(0, 2.5e5)
    
def fig2a(plt, cdf):
    """Fastplot callback for generating Fig.2a of the paper.
    Amount of packets per sender in 1 month.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    cdf : pandas.DataFrame
        packets per senders over a month
    """
    plt.plot(cdf.index, cdf.values, label='Unfiltered')
    plt.vlines(10, -.02, 1.02, color='r', linestyle='--', 
               label='Filtering threshold')
    plt.grid(linestyle='--')
    plt.xscale('log')
    plt.ylabel('ECDF')
    plt.xlabel('Monthly packets')
    plt.legend(loc = 'lower right')
    plt.xlim(.9, np.max(cdf.index)+2)
    plt.ylim(-.02, 1.02)
    
def fig2b(plt, cdf, cdf_f):
    """Fastplot callback for generating Fig.2b of the paper.
    Cumulative number of senders over time.
    
    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    cdf : pandas.DataFrame
        cumulative sum of senders over time unfiltered
    cdf_f : pandas.DataFrame
        cumulative sum of senders over time filtered over 30 days
    """
    plt.plot(cdf, linewidth=.8, marker='o', markersize=4, label='Unfiltered')
    plt.plot(cdf_f, linewidth=.8, marker='s', markersize=4, label='Filtered')
    plt.grid(linestyle='--')
    plt.yscale('log')
    plt.xlabel('$\Delta T$ [day]')
    plt.ylabel('Distinct IP addresses')
    plt.xticks([0, 4, 9, 14, 19, 24, 29], [1, 5, 10, 15, 20, 25, 30])
    plt.legend()
    plt.xlim(-.5, 29.5)
    
def fig8a(plt, stretchoid):
    """Fastplot callback for generating Fig.8a of the paper.
    Stretchoid activity pattern.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    stretchoid : pandas.DataFrame
        sequence of packets per source IP belonging to Stretchoid GT class
    """
    plt.scatter(stretchoid.index, stretchoid.tkn, s=.01, marker='o')
    plt.grid(linestyle='--')
    plt.xlabel('Time [day]')
    plt.ylabel('Sender')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlim(datetime.strptime('2021-03-02:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-03-31:00', '%Y-%m-%d:%H'))  
    plt.yticks([0, 100, 200, 300, 400, 500, 600, 700])
    plt.ylim(0, 700)
    
def fig8b(plt, en_um):
    """Fastplot callback for generating Fig.8b of the paper.
    Engin-Umich activity pattern.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    en_um : pandas.DataFrame
        sequence of packets per source IP belonging to Engin-Umich GT class
    """
    plt.scatter(en_um.index, en_um.tkn, s=1, marker='o')
    plt.grid(linestyle='--')
    plt.xlabel('Time [day]')
    plt.ylabel('Sender')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlim(datetime.strptime('2021-03-02:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-03-31:00', '%Y-%m-%d:%H'))   
    plt.ylim(-.5, 30)
    
def fig5(plt, gs_train_window):
    """Fastplot callback for generating Fig.5 of the paper.
    Impact of training window length.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    gs_train_window : pandas.DataFrame
        results of the experments abount the training window lenght
    """
    plt.plot(gs_train_window.training_days, gs_train_window.ip_set, 
             marker='o', color='k')
    plt.grid(linestyle='--')
    plt.ylim(20, 105)
    plt.xlim(.5, 30.5)
    plt.xticks([1, 5, 10, 20, 30])
    plt.xlabel('Training window size')
    plt.ylabel('Coverage [\%]')
    
def fig6(plt, knn_accs):
    """Fastplot callback for generating Fig.6 of the paper.
    Impact of k on the k-NN classifier.
    
    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    knn_accs : dict
        results of the experiments for the impact of classifier k
    """
    plt.plot(knn_accs['service_x'], knn_accs['service_y'], 
             marker='o', markersize=3, label='Per-service languages')
    plt.plot(knn_accs['auto_x'], knn_accs['auto_y'], 
             marker='s', markersize=3, label='Auto-defined languages')
    plt.plot(knn_accs['single_x'], knn_accs['single_y'], 
             marker='*', markersize=4, label='Single language')
    plt.xlabel('$k$')
    plt.legend()
    plt.ylabel('$k$-NN accuracy')
    plt.grid(linestyle='-.')
    plt.xticks([1, 3, 7, 17, 25, 35])
    plt.ylim(.3)
    
def fig7a1(plt, heatmaps, Vs, Cs):
    """Fastplot callback for generating the first part of Fig.7a of the paper.
    Auto-defined models, grid search through accuracy.
    
    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    heatmaps : list
        heatmaps resulting from the grid search. Knn classifier accuracy
    Vs : list
        embedding sizes Vs tested during the grid search
    Cs : list
        context window sizes Cs tested during the grid search
    """
    plt.gca().invert_yaxis()
    ax = sns.heatmap(heatmaps[1], cmap='Blues', annot=True, 
                     cbar_kws={'label':'Avg. accuracy'}, vmin=.83, vmax=.96)

    plt.ylabel('Embeddings size $V$')
    plt.xlabel('Context window $c$')
    plt.yticks([x+.5 for x in range(len(Vs))], Vs, rotation=0)
    plt.xticks([x+.5 for x in range(len(Cs))], Cs)
    ax.invert_yaxis()
    
def fig7a2(plt, heatmaps_time, Vs, Cs):
    """Fastplot callback for generating the second part of Fig.7a of the paper.
    Auto-defined models, grid search through model training runtime.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    heatmaps_time : list
        heatmaps resulting from the grid search. Training runtimes
    Vs : list
        embedding sizes Vs tested during the grid search
    Cs : list
        context window sizes Cs tested during the grid search
    """
    plt.gca().invert_yaxis()
    ax = sns.heatmap(heatmaps_time[1], cmap='Blues', annot=True, 
                     cbar_kws={'label':'Runtime [hour]'}, vmin=0, vmax=5)

    plt.ylabel('Embeddings size $V$')
    plt.xlabel('Context window $c$')
    plt.yticks([x+.5 for x in range(len(Vs))], Vs, rotation=0)
    plt.xticks([x+.5 for x in range(len(Cs))], Cs)
    ax.invert_yaxis()
    
def fig7b1(plt, heatmaps, Vs, Cs):
    """Fastplot callback for generating the first part of Fig.7b of the paper.
    Per-service models, grid search through accuracy.
    
    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    heatmaps : list
        heatmaps resulting from the grid search. Knn classifier accuracy
    Vs : list
        embedding sizes Vs tested during the grid search
    Cs : list
        context window sizes Cs tested during the grid search
    """
    plt.gca().invert_yaxis()
    ax = sns.heatmap(heatmaps[0], cmap='Blues', annot=True, 
                     cbar_kws={'label':'Avg. accuracy'}, vmin=.83, vmax=.96)

    plt.ylabel('Embeddings size $V$')
    plt.xlabel('Context window $c$')
    plt.yticks([x+.5 for x in range(len(Vs))], Vs, rotation=0)
    plt.xticks([x+.5 for x in range(len(Cs))], Cs)
    ax.invert_yaxis()
    
def fig7b2(plt, heatmaps_time, Vs, Cs):
    """Fastplot callback for generating the second part of Fig.7b of the paper.
    Per-service models, grid search through accuracy.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    heatmaps_time : list
        heatmaps resulting from the grid search. Training runtimes
    Vs : list
        embedding sizes Vs tested during the grid search
    Cs : list
        context window sizes Cs tested during the grid search
    """
    plt.gca().invert_yaxis()
    ax = sns.heatmap(heatmaps_time[0], cmap='Blues', annot=True, 
                     cbar_kws={'label':'Runtime [hour]'}, vmin=0, vmax=5)

    plt.ylabel('Embeddings size $V$')
    plt.xlabel('Context window $c$')
    plt.yticks([x+.5 for x in range(len(Vs))], Vs, rotation=0)
    plt.xticks([x+.5 for x in range(len(Cs))], Cs)
    ax.invert_yaxis()
    
def fig9(plt, ncs, mods):
    """Fastplot callback for generating Fig.9 of the paper.
    Impact of k' in cluster detection.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    ncs : list
        detected number of clusters per tested k'
    mods : list
        graph modularity with the clusters as partition per tested k'
    """
    ax1 = plt.gca()
    ax1.plot(range(1, 15), ncs, color='r', marker='o')
    ax2 = ax1.twinx()
    ax2.plot(range(1, 15), mods, color='b', linestyle='--', marker='s')
    ax1.set_xticks([x for x in range(1, 15)])
    ax1.set_yscale('log')
    ax2.set_xticks([x for x in range(1, 15)])
    ax1.set_xlim(.8, 14.2)
    ax2.set_xlim(.8, 14.2)
    ax1.set_xlabel("$k'$")
    ax1.set_ylabel('Number of Clusters', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('Modularity', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax1.grid(linestyle='-.')
    ax2.grid(linestyle='-.')
    plt.tight_layout()
    
def fig10(plt, shs):    
    """Fastplot callback for generating Fig.10 of the paper.
    Average silhouette of points within the found clusters.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    shs : pandas.DataFrame
        silhouette plot per cluster
    """
    plt.plot(shs.sh.values, linewidth=1)
    plt.hlines(0, 0, shs.shape[0], linestyle='-.', linewidth=.8)
    plt.xlabel('Cluster')
    plt.ylabel('Avg. Silhouette')
    cen = np.where(shs.index.isin([5, 28, 33, 34, 34, 39, 42, 44]))[0]
    cen_sh = shs.iloc[cen].sh.values
    shadows = np.where(shs.index.isin([25, 29, 37]))[0]
    shadows_sh = shs.iloc[shadows].sh.values
    plt.xticks([])
    plt.scatter(cen, cen_sh, s=15, color='r', marker='o', label='Censys')
    plt.scatter(shadows, shadows_sh, s=15, color='b', 
                marker='s', label='Shadowserver')
    plt.legend()
    
def fig11(plt, clusters, tick):
    """Fastplot callback for generating Fig.11 of the paper.
    Activity patterns of Censys sub-clusters.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    clusters : pandas.DataFrame
        division of the ips in clusters
    tick : pandas.DataFrame
        list of x-axis ticks for the plot
    """
    ax1 = plt.gca()
    for c in clusters.C.unique():
        ax1.scatter(clusters[clusters.C==c].index, 
                    clusters[clusters.C==c].tkn, s=.01, marker='o')
    ax1.grid(linestyle='--')
    
    ax1.set_xlabel('Time [day]')
    ax1.set_ylabel('Sender')
    ax1.set_yticks([])
    ax2 = ax1.twinx();
    ax2.set_ylabel('Community')
    ax2.set_yticks(tick.x)
    ax2.set_yticklabels([f'C{i}' for i in tick.index])
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlim(datetime.strptime('2021-03-03:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-03-31:00', '%Y-%m-%d:%H'))   
    plt.ylim(-1, 128)
    
def fig12(plt, clusters):
    """Fastplot callback for generating Fig.12 of the paper.
    Activity patterns of Shadowserver sub-clusters.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    clusters : pandas.DataFrame
        sequence of packets per source IP belonging to the provided clusters
    """
    plt.rcParams.update({'font.size': 14})
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    for c in [25, 29, 37]:
        temp__ = clusters[clusters.C == c]
        ax1.scatter(temp__.index, temp__.tkn, s=.05, marker='o', label=f'C{c}')
    plt.grid(linestyle='--')
    ax1.set_ylim(-1, 114)
    ax2.set_ylim(-1, 114)
    ax1.set_xlabel('Time [day]')
    ax1.set_ylabel('Sender')
    ax2.set_ylabel('Cluster')
    
    ax2.set_yticks([30, 61+12, 61+36+8])
    ax2.set_yticklabels([f'C{i}' for i in [25, 29, 37]])
    
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator, 
        formats=['%d', '%d', '%d', '%d', '%d', '%d'])
    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    plt.xticks([datetime.strptime('2021-03-03', '%Y-%m-%d'), 
                datetime.strptime('2021-03-04', '%Y-%m-%d'),
                datetime.strptime('2021-03-05', '%Y-%m-%d'),
                datetime.strptime('2021-03-06', '%Y-%m-%d')])
    plt.xlim(datetime.strptime('2021-03-03', '%Y-%m-%d'),
             datetime.strptime('2021-03-06', '%Y-%m-%d'))   

def plot_censys_jaccard(plt, jacc):
    """Fastplot callback for generating the jaccard heatmap. It represents the
    jaccard index between the ports contacted by the censys found sub-clusters

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    jacc : pandas.DataFrame
        jaccard matrix shaped `(n_clusters , n_clusters)`
    """
    to_r = {x: f'C{x}' for x in jacc.index}
    sns.heatmap(jacc.astype(float).rename(columns=to_r, index=to_r), 
                cmap='Blues', cbar_kws={'label':'Jaccard Index'})
    plt.yticks(rotation=0)
    plt.xlabel('Censys sub-clusters')
    plt.ylabel('Censys sub-clusters')
    
def plot_generic_pattern(plt, C_):    
    """Fastplot callback for plotting the activity patterns of a generic 
    provided cluster

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    C_ : pandas.DataFrame
        activity patterns of IPs partition in clusters to test
    """
    sns.heatmap(C_, cmap='magma', norm=LogNorm(), cbar_kws={'label':'Packets'})
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Port')
    plt.ylabel('IP Sources')
    
def plot_port_pattern(plt, clusters_):
    """Fastplot callback for plotting the pattern of the ports contacted by
    IPs belonging to the provided clusters.

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    clusters_ : pandas.DataFrame
        set of destination ports timeseries per source IP
    """
    plt.scatter(clusters_.index, clusters_.tkn, s=1, marker='o')
    plt.grid(linestyle='--')
    plt.xlabel('Time [day]')
    plt.ylabel('Sender')
    plt.xticks(rotation=30)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)