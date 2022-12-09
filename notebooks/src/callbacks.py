import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.colors import LogNorm
import matplotlib as mpl
from matplotlib import cm

###############################################################################
# Jupyter-notebooks 01-darknet-overview.ipynb
###############################################################################
def portRanking(plt, pkts, top):
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
    
def darknetPatterns(plt, tday):
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
    
def filterECDF(plt, cdf):
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
    
def filterCoverage(plt, cdf, cdf_f):
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
    plt.xlabel('$Delta T$ [day]')
    plt.ylabel('Distinct IP addresses')
    plt.xticks([0, 4, 9, 14, 19, 24, 29], [1, 5, 10, 15, 20, 25, 30])
    plt.legend()
    plt.xlim(-.5, 29.5)
    
def stretchoidPattern(plt, stretchoid):
    """Fastplot callback for generating Fig.10a of the paper.
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
    
def enginumichPattern(plt, en_um):
    """Fastplot callback for generating Fig.10b of the paper.
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

def ground_truth_heatmap(plt, pivot):
    """Generate the heatmap with the fraction of packets per service for each
    ground truth class
    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    pivot : pandas.DataFrame
        data to plot
    """
    cols = dict()
    rows = dict()
    for c in pivot.index:
        c1 = c
        if c == 'others': c = 'Others'
        elif c== 'mail': c = 'Mail'
        elif c== 'kerberos': c = 'Kerberos'
        elif c== 'netbios': c = 'NetBIOS'
        elif c== 'telnet': c = 'Telnet'
        elif c== 'icmp': c = 'ICMP'
        elif c== 'netbios-smb': c = 'NetBIOS-SMB'
        else: c = c.upper()
        cols[c1] = c
    for r in pivot.columns:
        if r == 'mirai':
            r1 = 'mirai-like'
        else:
            r1 =r
        rows[r] = r1.capitalize()
    sns.heatmap(pivot.rename(index=cols, columns=rows), 
                cmap='coolwarm', norm=LogNorm(), 
                cbar_kws={'label':'Fraction of daily packets'})

    plt.xlabel('Ground truth class')
    plt.ylabel('Service')
###############################################################################
# Jupyter-notebooks 02-baseline.ipynb
###############################################################################
def clusteringBaseline(plt, df):
    """Heatmap of ground truth w.r.t. assigned labels after supervised 
    clustering

    Parameters
    ----------
    plt : matplotlib.pyplot
        matplotlib instance for fastplot callback
    df : pandas.DataFrame
        data to plot
    """
    new_idx = dict()
    for x in df.index:
        if x == 'mirai':
            x1 = 'mirai-like'
        else: 
            x1 = x
        new_idx[x] = x1.capitalize()

    sns.heatmap(df.rename(index=new_idx), cmap='coolwarm', 
                cbar_kws={'label':'GT points per cluster [\%]'}, vmin=0)

    plt.xlabel('Assigned Cluster')
    plt.ylabel('True Label')

###############################################################################
# Jupyter-notebooks 03-gridsearch.ipynb
###############################################################################
def plotTrainingWindow(plt, gs_train_window):
    """Fastplot callback for generating Fig.7 of the paper.
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
    plt.ylabel('Coverage [%]')

def plotTopN(plt, perclassx, perclassy, macrosx, macrosy):   
    plt.scatter(perclassx, perclassy, s=5, label='Per class', color='k')
    plt.plot(macrosx, macrosy, markersize=5, marker='s', 
             label='Average', color='k')
    plt.xscale('log')
    plt.grid(linestyle='-.')
    plt.legend()
    plt.xlabel('Top-$n$ ports')
    plt.ylabel('Macro F-Score')
    plt.xticks([1, 10, 50, 100, 500, 2500, 10000],  
               [1, 10, 50, 100, 500, 2500, 10000])
    plt.xlim(0.85, 11500)
    plt.ylim(-.05, 1.05)
    plt.legend()
    plt.grid(linestyle='-.')
    
def plotKofKnn(plt, x, y1, y2, y3, y4):
    plt.plot(x, y1, marker='o', markersize=4, label='DKS')
    plt.plot(x, y2, marker='s', markersize=4, label='SS')
    plt.plot(x, y3, marker='^', markersize=4, label='AS')
    plt.plot(x, y4, marker='*', markersize=4, label='HS')
    plt.xlabel('$k$')
    plt.ylabel('Avg. $k$-NN Macro F-Score')
    plt.xticks([1, 3, 5, 7, 17, 25, 35], [1, 3, 5, 7, 17, 25, 35])
    plt.xlim(0, 40)
    plt.ylim(0, 1)
    plt.grid(linestyle='-.')
    plt.legend(ncol=2)
    
def plotGridsearchFscore(plt, _heatmap):
    ax = plt.gca()
    sns.heatmap(_heatmap, annot=True, cmap='Blues', ax=ax, vmin=.6, vmax=.9, 
                cbar_kws={'label':'Avg. Macro F-Score'})
    ax.set_xlabel('Embedding size $E$')
    ax.set_ylabel('Context window size $C$')
    ax.invert_yaxis()
    
def plotGridsearchRuntimes(plt, _heatmap):
    ax = plt.gca()
    sns.heatmap(_heatmap, annot=True, cmap='Blues', ax=ax, vmin=1, vmax=22, 
                cbar_kws={'label':'Training runtime [min]'})
    ax.set_xlabel('Embedding size $E$')
    ax.set_ylabel('Context window size $C$')
    ax.invert_yaxis()

def densityPlot(plt, df):
    ax = plt.gca()
    sns.kdeplot(x=df.x, y=df.y, cmap="Blues", shade=True, gridsize=500, ax=ax)
    ax.set_xlabel('$1^{st}$ t-SNE component')
    ax.set_ylabel('$2^{nd}$ t-SNE component')

###############################################################################
# Jupyter-notebooks 04-clustering.ipynb
###############################################################################
def plotKnnGraphHeuristic(plt, ncs_linear, mods_linear):
    """Fastplot callback for generating Fig.11 of the paper.
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
    ax1.plot(range(1, 15), ncs_linear, color='r', marker='o')
    ax2 = ax1.twinx()
    ax2.plot(range(1, 15), mods_linear, color='b', linestyle='--', marker='s')
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

def sh_sources(plt, knn_ecdf, kmeans_ecdf, spectral_ecdf, hierarchical_ecdf):
    plt.plot(knn_ecdf.sh, knn_ecdf.coverage, color='k', label='3-NN Graph')
    plt.plot(kmeans_ecdf.sh, kmeans_ecdf.coverage, color='r', label='kMeans')
    plt.plot(spectral_ecdf.sh, spectral_ecdf.coverage, color='g', label='Spectral')
    plt.plot(hierarchical_ecdf.sh, hierarchical_ecdf.coverage, color='b', label='Hierarchical')
    plt.xlabel('Silhouette')
    plt.ylabel('Sources ECDF')
    plt.legend()
    plt.grid(linestyle='-.')
    plt.xlim(-1, 1.3)

def scatterplot_sh(plt, scatters):
    cm = plt.cm.get_cmap('cool')
    xy = np.arange(-1, 1, .1)
    sc = plt.scatter(scatters.ip, scatters.sh, s=8, color='k')
    plt.xlabel('Source per cluster')
    plt.ylabel('Avg. $Sh$ per cluster')
    plt.xscale('log')
    #plt.yscale('log')
    plt.hlines(0, 1, 1e4, color='k', linestyle='--')
    plt.grid(linestyle='-.')

def shplot_(plt, sh_df):
    censys = [7, 16, 20, 48, 52, 55, 60]
    shadowserver = [3, 10, 12, 32, 34, 50, 63, 64, 65, 66, 68, 69]
    mirai = [39, 45, 36]
    unknown = [59, 30]
    plt.plot(sh_df.index, sh_df.sh, color='k')
    temp = sh_df[sh_df.C.isin(censys)]
    plt.scatter(temp.index, temp.sh, marker='o', s=30, alpha=1, color='r', label = 'Censys')
    temp = sh_df[sh_df.C.isin(shadowserver)]
    plt.scatter(temp.index, temp.sh, marker='s', s=30, alpha=1, color='g', label= 'Shadowserver')
    temp = sh_df[sh_df.C.isin(mirai)]
    plt.scatter(temp.index, temp.sh, marker='^', s=30, alpha=1, color='b', label= 'Mirai-like')
    temp = sh_df[sh_df.C.isin([unknown[0]])]
    plt.scatter(temp.index, temp.sh, marker='*', s=100, alpha=1, color='orange', label='C59')
    temp = sh_df[sh_df.C.isin([unknown[1]])]
    plt.scatter(temp.index, temp.sh, marker='X', s=80, alpha=1, color='y', label='C30')
    plt.legend(ncol=2)
    
    plt.hlines(0, 0, 80, linestyle='--', color='k')
    plt.xlabel('Cluster rank')
    plt.ylabel('Avg. Silhouette')
    plt.grid(linestyle='--')
    plt.ylim(-1.1, 1)

###############################################################################
# Jupyter-notebooks 05-clusters-inspection.ipynb
###############################################################################
def clusterdensityPlot(plt, tsne_df, C):
    to_plot = tsne_df[tsne_df.C == C]
    others = tsne_df[tsne_df.C != C]
    plt.scatter(others.x, others.y, color='k', s=.1, alpha=.6)
    if to_plot.shape[0]<100:
        plt.scatter(to_plot.x, to_plot.y, color='r', s=50, marker='*')
    else:
        plt.scatter(to_plot.x, to_plot.y, color='r', s=.1, marker='*')
    plt.xlabel('$1^{st}$ t-SNE component')
    plt.ylabel('$2^{nd}$ t-SNE component')
    
def clusterActivityPattern(plt, cluster_trace, C):
    plt.scatter(cluster_trace.index, cluster_trace.token, s=.02, color='k')
    plt.grid(linestyle='--')
    plt.xlabel('Time [day]')
    plt.ylabel('Sender')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlim(datetime.strptime('2021-03-02:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-04-01:00', '%Y-%m-%d:%H'))   

def Top4Traffic(plt, cluster_trace, top5, C):
    top4 = top5[:-1]
    top5_trace = cluster_trace[cluster_trace.pp.isin(top4.pp)]

    for port in top4.pp:
        top5_ = top5_trace[top5_trace.pp == port]
        top5_ = top5_.resample('H').agg({'pp':'count'})

        plt.plot(top5_, label=port, linewidth=1)
    plt.yscale('log')
    plt.legend(ncol=2) 
    plt.grid(linestyle='--')
    plt.xlabel('Time [hour]')
    plt.ylabel('Cluster traffic [packets]')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlim(datetime.strptime('2021-03-20:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-04-01:00', '%Y-%m-%d:%H'))   
    
    
def portHeatmap(plt, temp):
    pivotted = temp.pivot_table(values='pkts', columns='iptoken', index='ptoken', aggfunc='sum')
    sns.heatmap(pivotted, norm=LogNorm(), cbar_kws={'label':'Packets'})
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Port rank')
    plt.ylabel('Senders')

def portPattern(plt, temp):
    plt.scatter(temp.index, temp.ptoken, s=.02, color='k')
    plt.grid(linestyle='--')
    plt.xlabel('Time [day]')
    plt.ylabel('Port rank')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlim(datetime.strptime('2021-03-02:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-04-01:00', '%Y-%m-%d:%H'))  


def clusterShadowserverPatternFinal(plt, cluster_trace):  
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(0, cluster_trace.C.unique().shape[0]+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    ax1 = plt.gca()
    for c in cluster_trace.C.unique():
        temp = cluster_trace[cluster_trace.C==c]
        ax1.scatter(temp.index, temp.iptoken, s=.5, marker='o')
    ax1.grid(linestyle='--')
    
    ax1.set_xlabel('Time [day]')
    ax1.set_ylabel('Sender')
    
    h_fmt = mdates.DateFormatter('%d')
    hours = mdates.HourLocator(interval = 24)
    plt.gca().xaxis.set_major_locator(hours)
    plt.gca().xaxis.set_major_formatter(h_fmt)
    ax1.set_xlim(datetime.strptime('2021-03-03:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-03-08:00', '%Y-%m-%d:%H'))   
    bounds = bounds+0.5
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=bounds, label='Cluster')
    cbar.set_ticklabels([f'C{c}' for c in cluster_trace.C.unique()])
    
    

def clusterCensysPatternFinal(plt, cluster_trace):
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(0, cluster_trace.C.unique().shape[0])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    ax = plt.gca()
    for c in cluster_trace.C.unique():
        temp = cluster_trace[cluster_trace.C == c]
        if c == 7:
            ax.scatter(temp.index, temp.iptoken, s=.000003)
        else:
            ax.scatter(temp.index, temp.iptoken, s=.01)
    ax.grid(linestyle='--')
    ax.set_xlabel('Time [day]')
    ax.set_ylabel('Sender')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    ax.set_xlim(datetime.strptime('2021-03-20:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-03-30:12', '%Y-%m-%d:%H'))   
    bounds = bounds+0.5
    ax.set_ylim(0, cluster_trace.ip.unique().shape[0])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=bounds, label='Cluster')
    cbar.set_ticklabels([f'C{c}' for c in cluster_trace.C.unique()])

def clusterCensysPatternFinal(plt, cluster_trace):
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(0, cluster_trace.C.unique().shape[0]+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    ax = plt.gca()
    for c in cluster_trace.C.unique():
        temp = cluster_trace[cluster_trace.C == c]
        if c == 7:
            ax.scatter(temp.index, temp.iptoken, s=.000003)
        else:
            ax.scatter(temp.index, temp.iptoken, s=.1)
    ax.grid(linestyle='--')
    ax.set_xlabel('Time [day]')
    ax.set_ylabel('Sender')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    ax.set_xlim(datetime.strptime('2021-03-20:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-03-30:12', '%Y-%m-%d:%H'))   
    bounds = bounds+0.5
    ax.set_ylim(0, cluster_trace.ip.unique().shape[0])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=bounds, label='Cluster')
    cbar.set_ticklabels([f'C{c}' for c in cluster_trace.C.unique()])


def clusterPortPatternFinal(plt, cluster_trace):
    cmap = plt.cm.jet 
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(0, cluster_trace.ip.unique().shape[0])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    for ip in cluster_trace.ip.unique():
        temp = cluster_trace[cluster_trace.ip == ip]
        plt.scatter(temp.index, temp.ptoken, s=1)
    plt.grid(linestyle='--')
    plt.xlabel('Time [day]')
    plt.ylabel('Port rank')
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlim(datetime.strptime('2021-03-19:00', '%Y-%m-%d:%H'),
             datetime.strptime('2021-04-01:00', '%Y-%m-%d:%H'))   
    plt.ylim(0, cluster_trace.ptoken.unique().shape[0])
    
    bounds = bounds+0.5
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=bounds, label='Sender')
    cbar.set_ticklabels(['$IP_{'+str(k)+'}$' for k,v in enumerate(cluster_trace.ip.unique())])