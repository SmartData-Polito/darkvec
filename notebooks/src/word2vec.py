from gensim.models import Word2Vec as W2V 
from multiprocessing import cpu_count
import pandas as pd

class Word2Vec():
    def __init__(self, c=25, e=50, epochs=20, mname='sample', method=None):
        self.context_window = c
        self.embedding_size = e
        self.epochs = epochs
        self.mname = mname
        self.model = None

    def train(self, corpus, save=False):
        self.model = W2V(sentences=corpus, size=self.embedding_size, 
                         window=self.context_window, iter=self.epochs, 
                         workers=cpu_count(), min_count=0, sg=1, negative=5, 
                         sample=0, seed=15)
        
        if save:
            self.model.save(f'{self.mname}.model')

    def load_model(self):
        self.model = W2V.load(f'{self.mname}.model')

    def get_embeddings(self, ips, labels):
        # Return a dataframe with the provided IPs as index
        # If provided also labels they are set as last column named `class`
        # Labels must be provided as another pandas dataframe with a column
        # named `ip` and a column named `class`
        embeddings = [self.model.wv.__getitem__(x) for x in ips]
        embeddings = pd.DataFrame(embeddings, index=ips)
        embeddings = embeddings.reset_index().rename(columns={'index':'ip'})\
                               .merge(labels, on='ip', how='left')\
                               .set_index('ip')
        return embeddings
    
    def update(self, corpus, save=False):
        self.model.build_vocab(corpus, update=True, trim_rule=None)
        self.model.train(corpus, total_examples=self.model.corpus_count, 
                         epochs=self.epochs)
        if save:
            self.model.save(f'{self.mname}.model')