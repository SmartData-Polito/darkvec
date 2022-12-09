from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report as report
import numpy as np


class KnnClassifier():
    def __init__(self, embeddings, k):
        self.embeddings = embeddings
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.y_pred = None
        self.labels = self.embeddings[
            self.embeddings['class']!='unknown']['class'].unique()
        self.knn = KNN(n_neighbors=k+1, metric='cosine', n_jobs=10)
        self.split_test_train()
        
    def split_test_train(self):
        # Remove the 'class' columns
        if 'class' in self.embeddings.columns:
            temp = self.embeddings.drop(columns=['class'])
        else:
            temp = self.embeddings.copy()
            
        self.x_train = temp.to_numpy()
        self.y_train = self.embeddings['class'].values

        temp  = self.embeddings[self.embeddings['class']!='unknown']
        self.x_test = temp.drop(columns=['class'])
        self.x_test = self.x_test.to_numpy()
        self.y_test = temp['class'].values
        
    def majority_voting(self, x):
        lab, freqs = np.unique(x, return_counts=True)
        if lab.shape[0] == 1:
            return np.array(lab[0])
        else:
            if freqs[0]==freqs[1]:
                return np.array('n.a.')
            else:
                return np.array(lab[0])
        
    def fit_predict(self):
        self.knn.fit(self.x_train, self.y_train)
        pred_idx = self.knn.kneighbors(self.x_test)[1][:, 1:]
        pred_lab = self.y_train[pred_idx]
        self.y_pred = np.asarray([self.majority_voting(i) for i in pred_lab])

    def get_report(self, output_dict=False, labels=None):
        if isinstance(labels, type(None)):
            labels = self.labels
        
        knn_report = report(self.y_test, self.y_pred, 
            output_dict=output_dict, labels=self.labels)
        
        return knn_report 