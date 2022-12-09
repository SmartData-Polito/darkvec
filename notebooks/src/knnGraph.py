import pandas as pd
import networkx as nx
import community as gma
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
from math import e


def linear_weighting(_x):
    return 1-np.abs(_x)
def negexp_weighting(_x, a, b):
    return a*e**(-np.abs(_x)/b)
def poly_weighting(_x, deg):
    return (1-np.abs(_x))**deg
def gauss_weighting(_x, a, b, c):
    return a*e**(-((np.abs(_x)-b)**2)/(2*c**2))


class KnnGraph():
    """Create the knn graph starting from the provided embeddings. 
    The knn graph G = (N, V) where N is a set of nodes and V is a set of 
    vertices is made so that a link exists among to nodes if they belong to the 
    same k nearest neighborhood. The weights are the cosine similarity among 
    the nodes.

    Then it run the Louvain algorithm on the generated graph determining the
    best partition of clusters maximizing the graph modularity.

    Parameters
    ----------
    exp_id : str
        experiment identifier. It is the name of the experiment in the model/
        folder
    day : str
        considered day of the experiment. The day is in the form YYYYMMDD, by 
        default '20210302'
    graph_gen : bool, optional
        if True, a new knn graph is created, otherwise it loads the existing 
        graph from the experiment folder, by default False
    k : int, optional
        number of nearest neighbors used for the label assignment, by default 4
    embeddings : pandas.DataFrame, optional
        embeddings dataframe shaped (N, E+1). Bing independent from the ground
        truth, the `class` column is not required, so the addigional one is the 
        `ip` column, by default None
    ips : list, optional
        list of the IPs ordered according to the embedding dataframe, by 
        default None
    labels : list, optional
        list of the labels ordered according to the embedding dataframe, by 
        default None
    weighting_function : tuple
        function used to assign edge weights from the distance and parameters.
    
    Attributes
    ----------
    embeddings : pandas.DataFrame
        embeddings dataframe shaped (N, E+1). Bing independent from the ground
        truth, the `class` column is not required, so the addigional one is the 
        `ip` column, by default None
    ips : list
        list of the IPs ordered according to the embedding dataframe, by 
        default None
    labels : list
        list of the labels ordered according to the embedding dataframe, by 
        default None
    mod : float
        modularity value of the partition provided (`clusters` attribute)
    clusters : dict
        clusters partition from the application of the Louvain algorithm in the
        form `{source_ip:cluster_id}`
    nc : int
        number of found clusters through the Louvain algorithm
    k : int
        number of nearest neighbors used for the label assignment, by default 4
    gname : str
        identifier of the graph in the form `exp_id.day.k`
    G : networkx.classes.graph.Graph
        initialized k-nearest-neighbor graph
    graph_gen : bool
        if True, a new knn graph is created, otherwise it loads the existing 
        graph from the experiment folder
    weighting_function : tuple
        function used to assign edge weights from the distance and parameters.

    Methods
    -------
    get_knn_pos():
        get the position and the distance of the k nearest neighbors wrt. 
        target samples.
    load_graph():
        load an existing graph
    create_graph(pos, dist):
        create the k nearest neighbor graph from the provided embeddings
    fit_predict(n_it):
        run the Louvain algorithm and get the clusters partition
    """
    def __init__(self, exp_id, day, graph_gen=False, k=4, embeddings=None, 
        ips=None, labels=None, weighting_function=[(linear_weighting)]):
        self.embeddings = embeddings
        self.labels = labels
        self.ips = ips
        
        self.mod = None # Graph modularity
        self.clusters = None # Found clusters
        self.nc = None # Number of found clusters
        self.k = k # Used k for the knn graph
        
        self.graph_gen = graph_gen
        self.gname = f'{exp_id}.{day}.{self.k}'
        
        self.wfunc = weighting_function
        #self.gname = f'G{self.gname}.gexf'

        if not graph_gen:
            self.G = self.load_graph()
        else:
            pos, dist = self.get_knn_pos()
            self.G = self.create_graph(pos, dist)

    def get_knn_pos(self):
        """Exploit the classifier retrieving the position of the k nearest 
        neighbors and the similarity with the target sample. The similarity is
        1-| d |, where d is the cosine distance among the target sample and 
        its k nearest neighbors.

        Returns
        -------
        tuple
            the position of the k nearest neighbors and the distance with the
            target sample
        """
        X = self.embeddings.copy()
        y = self.labels
        
        try: 
            y == None
            y = np.zeros((X.shape[0]))
        except: pass
        if 'class' in X.columns:
            X = X.drop(columns=['class'])
        knn = KNN(n_neighbors=self.k+1, metric='cosine', n_jobs=-1)
        knn.fit(X, y)
        dist, pos = knn.kneighbors(X)
        
        if type(self.wfunc) == type(lambda x:x):
            dist = self.wfunc(dist)
        else:
            dist = self.wfunc[0](dist, *self.wfunc[1:])
        return pos, dist
        
    def load_graph(self, gname):
        """Load an existing graph from the `model/exp_id/day/graph` folder.

        Returns
        -------
        networkx.classes.graph.Graph
            initialized k-nearest-neighbor graph
        """
        exp_id, day, k = self.gname.split('.')
        G = nx.read_gexf(f'{gname}.gexf')
        return G
        
    def create_graph(self, pos, dist):
        """Create the knn graph starting from the provided embeddings.
        The knn graph G = (N, V) where N is a set of nodes and V is a set of
        vertices is made so that a link exists among to nodes if they belong
        to the same k nearest neighborhood.
        The weights are the cosine similarity among the nodes.

        Parameters
        ----------
        pos : list
            position of the k nearest neighbors 
        dist : list
            distance of the k nearest neighbors with the target sample

        Returns
        -------
        networkx.classes.graph.Graph
            initialized k-nearest-neighbor graph
        """
        links = set()
        atts = {}
        links_att = {}
                
        for pairs in zip(pos, dist):
            entry = pairs[0]
            s_idx = entry[0]
            if type(self.labels) is np.ndarray:
                s_class = self.labels[s_idx]
            s_ip = self.ips[s_idx]
            if s_ip not in atts and type(self.labels) is np.ndarray:
                atts[s_ip] = s_class
            for i in range(entry.shape[0]-1):
                neigh = entry[1+i]
                d_idx = neigh
                if type(self.labels) is np.ndarray:
                    d_class = self.labels[d_idx]
                d_ip = self.ips[d_idx]
                if d_ip not in atts and type(self.labels) is np.ndarray:
                    atts[d_ip] = d_class
                link = (s_ip, d_ip)
                w = pairs[1][i+1]
                if link not in links:
                    links.add(link)
                if (s_ip, d_ip) not in links_att or (d_ip, s_ip) not in links_att:
                    links_att[link] = w

        links_df = pd.DataFrame(
            [{'s':k[0], 'd':k[1], 'weight':v} for k,v in links_att.items()])
        G = nx.from_pandas_dataframe(links_df, 
                                     source='s', 
                                     target='d', 
                                     edge_attr='weight')
        G.add_nodes_from(atts.keys() - set(G.nodes()))
        if type(self.labels) is np.ndarray:
            nx.set_node_attributes(G, 'gt_class', atts)

        return G
        
    def fit_predict(self, n_it = None, gname=None):
        """Run the Louvain algorithm oon the generated k nearest neighbor 
        graphs to obtain the clusters partition. Then compute the modularity
        and save the graph with the cluster id as node attribute. Note. the 
        graph is always saved

        Parameters
        ----------
        n_it : int, optional
            number of Louvain iteration. It will be used in the iterative 
            version of the knn graph, by default None
        """
        exp_id, day, k = self.gname.split('.')
        self.clusters = gma.best_partition(self.G,random_state=15)
        self.mod = gma.modularity(self.clusters, self.G)
        self.nc = len(set([v for k, v in self.clusters.items()]))
        nx.set_node_attributes(self.G, 'community', self.clusters)
        if n_it == None and self.graph_gen:
            if gname!=None:
                nx.write_gexf(self.G, 
                    f'{gname}.gexf')
                print(f'Graph {self.gname} saved')
        elif n_it != None and self.graph_gen:
            if gname!=None:
                nx.write_gexf(self.G, 
                    f'{gname}.gexf')
                print(f'Graph {self.gname} saved')