from sklearn.cluster import KMeans
from src.utils import split_train_test


class Kmeans():
    def __init__(self, dataset, k):
        """Implementation of the k-Means used in the completely sueprvised 
        clustering on the baseline case. The class takes in input the dataset
        and the number of clusters, then run the fitting and the cluster labels
        assignment

        Parameters
        ----------
        dataset : pandas.DataFrame
            (`N_samples x N_features + gt_class`) dataset to cluster
        k : int
            numer of clusters for the partition
        """
        self.dataset = dataset
        self.k = k
        self.kmeans = KMeans(n_clusters=self.k, random_state=15, 
                             algorithm='auto')
        self.X_train=None
        self.y_train=None
        self.X_test=None
        self.y_test=None
        self.y_true=None
    
    def fit(self, X_train=None):
        """Fit the k-Means classifier. If the X_train dataset is provided, the
        algorithm is fitted on it. Otherwise split the dataset into training one
        (full ground truth labels + unknown) and testing one (full ground truth
        labels + unknown if desired)

        Parameters
        ----------
        X_train : numpy.ndarray, optional
            (`N_samples x N_features + gt_class`) dataset used to fit the 
            k-Means, by default None
        """
        if X_train == None:
            self.X_train, self.y_train, self.X_test, self.y_test =\
                split_train_test(self.dataset, with_unknown=True)
            self.y_true = self.y_test
            X_train = self.X_train
            
        self.kmeans.fit(X_train)
            
    def predict(self, X_test=None):
        """After having fitted the dataset, run the algorithm and assign the
        cluster labels to the provided dataset. If `X_test` is not provided, 
        the dataset used during the fitting is used.

        Parameters
        ----------
        X_test : numpy.ndarray, optional
            (`N_samples x N_features + gt_class`) dataset to cluster, by
             default None

        Returns
        -------
        list
            assigned cluster labels
        """
        if X_test == None:
            X_test = self.X_test
            
        y_pred = self.kmeans.predict(X_test)
        
        return y_pred
    
    def fit_predict(self, X_train=None, X_test=None):
        """Fit the k-Means on the provided `X_train` and cluster `X_test` 

        Parameters
        ----------
        X_train : numpy.ndarray, optional
            (`N_samples x N_features + gt_class`) dataset used to fit the 
            k-Means, by default None
        X_test : numpy.ndarray, optional
            (`N_samples x N_features + gt_class`) dataset to cluster, by
             default None

        Returns
        -------
        list
            assigned cluster labels
        """
        self.fit(X_train)
        y_pred = self.predict(X_test)
        
        return y_pred