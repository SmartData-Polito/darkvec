"""
Implementation of the k-Nearest-Neighbors Graph with the Louvain algorithm 
application for cluster detection. 
The module builds a Graph from a set of embeddings. The nodes are the IPs, and 
the link among two nodes exists if they belongs to the same k-neighborhood. The 
edges weights are the cosine similarity among the nodes pairs.  
The Louvain algorithm is applied and the cluster id of each node is saved 
as the attribute `community`. 
"""

import pandas as pd
import networkx as nx
import community as gma
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np

class KnnGraph():
    """Implementation of the k-Nearest-Neighbors Graph with the Louvain 
    algorithm application for cluster detection. 
    
    The module builds a Graph from a set of embeddings. The nodes are the IPs, 
    and the link among two nodes exists if they belongs to the same 
    k-neighborhood. The edges weights are the cosine similarity among the nodes
    pairs.  
    
    The Louvain algorithm is applied and the cluster id of each node is saved 
    as the attribute `community`. 

    Parameters
    ----------
    graph_path : str, optional
        global path to create a directory named as `model_name` containing 
        the model, scalers and graphs, by default None
    graph_gen : bool, optional
        if True generate a new knn graph from scratch. If False load an 
        existing .gexf graph, by default False
    k : int, optional
        number of nearest neighbors to use during the kNN graph creation. If
        None the heuristic for k is performed, by default 4
    embeddings : numpy.ndarray, optional
        darkvec embeddings used to built the knn graph, by default None
    ips : list, optional
        set of IPs for which the embeddings must be generated, by default 
        None
    labels : list, optional
        ground truth labels of the `ips`, by default None

    Attributes
    ----------
    graph_path : str
        global path to create a directory named as `model_name` containing 
        the model, scalers and graphs.
    embeddings : numpy.ndarray
        darkvec embeddings used to built the knn graph.
    labels : list, optional
        ground truth labels of the `ips`.
    k : int
        number of nearest neighbors to use during the kNN graph creation.
    gname : str
        `Gknn.gexf` where `k` is the passed value.
    mod : float
        modularity value of the graph after the Louvain application.
    comms : dict
        detected communities. The keys are the `ips`, the values are the id 
        of the communities.
    nc : int
        number of distinct communities.
    G : networkx.classes.graph.Graph)
        k-Nearest-Neighbor Graph built from the generated darkvec embeddings.
    """
    def __init__(self, graph_path=None, graph_gen=False, k=4, embeddings=None, 
                 ips=None, labels=None):
        self.embeddings = embeddings
        self.graph_path = graph_path
        self.labels = labels
        self.ips = ips
        
        self.mod = None
        self.comms = None
        self.nc = None
        self.k = k
        
        self.gname = f'G{self.k}nn.gexf'

        if not graph_gen:
            self.G = self.load_graph()
        else:
            pos, dist = self.get_knn_pos(load_scaler=False, save_scaler=True)
            self.G = self.create_graph(pos, dist)        

    def get_knn_pos(self, load_scaler=False, save_scaler=False):
        """Find the *k* nearest neighbors of the provided IPs. Returns their 
        indices and the relative distances.

        Parameters
        ----------
        load_scaler : bool, optional
            If True load a pre-fitted MinMaxScaler (range[0, 1]) for the 
            embeddings distance. If False fit a new scaler, by default False
        save_scaler : bool, optional
            If True save the fitted scaler, by default False

        Returns
        -------
        tuple
            (numpy.ndarray, numpy.ndarray). Indices of the k nearest neighbors 
            of each embedding; distances between the embeddings and their k 
            nearest neighbors.
        """
        X = self.embeddings
        y = self.labels
        knn = KNN(n_neighbors=self.k+1, metric='cosine', n_jobs=-1)
        knn.fit(X, y)
        dist, pos = knn.kneighbors(X)
        
        dist = 1-np.abs(dist)

        return pos, dist
    
    def load_graph(self):
        """Load an existing .gexf graph.

        Returns
        -------
        networkx.classes.graph.Graph
            k-Nearest-Neighbor Graph built from the generated darkvec 
            embeddings.
        """
        G = nx.read_gexf(f"{self.graph_path}/{self.gname}")
        return G

    def create_graph(self, pos, dist):
        """Take the indices of the k  nearest neighbors of each embeddings and
        the distances between the embeddings and their k nearest neighbors. Use
        them to create a weighted k-Nearest-Neighbors Graph. The weights are 
        the cosine similarities among nodes (ips).

        Parameters
        ----------
        pos : numpy.ndarray
            Indices of the k nearest neighbors of each embedding.
        dist : numpy.ndarray
            Distances between the embeddings and their k nearest neighbors.

        Returns
        -------
        networkx.classes.graph.Graph
            k-Nearest-Neighbor Graph built from the generated darkvec 
            embeddings.
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
        G = nx.from_pandas_edgelist(links_df, 
                                     source='s', 
                                     target='d', 
                                     edge_attr='weight')
        
        G.add_nodes_from(atts.keys() - set(G.nodes()))
        if type(self.labels) is np.ndarray:
            nx.set_node_attributes(G=G, name='gt_class', values=atts)

        return G

    def fit_predict(self, save_graph=False):
        """Run the Louvain algorithm on the knn graph finding the best nodes 
        partition and compute the modularity.

        Parameters
        ----------
        save_graph : bool, optional
            if True save a .gexf file compatible with Gephi, by default False
        """
        self.comms = comms = gma.best_partition(self.G,random_state=15)
        self.mod = gma.modularity(comms, self.G)
        self.nc = len(set([v for k, v in comms.items()]))
        nx.set_node_attributes(G=self.G, name='community', values=comms)
        
        if save_graph:
            nx.write_gexf(self.G, f"{self.graph_path}/{self.gname}")
            print(f'Graph {self.gname} saved')
