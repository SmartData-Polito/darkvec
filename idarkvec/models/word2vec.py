from gensim.models import Word2Vec as W2V 
from multiprocessing import cpu_count
import pandas as pd
import numpy as np

class Word2Vec():
    """_summary_

    Parameters
    ----------
    c : int, optional
        _description_, by default 25
    e : int, optional
        _description_, by default 50
    epochs : int, optional
        _description_, by default 1
    source : _type_, optional
        _description_, by default None
    destination : _type_, optional
        _description_, by default None
    seed : int, optional
        _description_, by default 15
    """
    def __init__(self, c=25, e=50, epochs=1, source=None, destination=None, 
                                                                      seed=15):
        self.context_window = c
        self.embedding_size = e
        self.epochs = epochs
        self.seed = seed
        self.source = source
        self.destination = destination

        if type(source) != type(None):
            self.model = W2V.load(f'{self.source}.word2vec')
        else:
            self.model = None
                
    def train(self, corpus, save=False):
        """_summary_

        Parameters
        ----------
        corpus : _type_
            _description_
        save : bool, optional
            _description_, by default False
        """
        print(f'[WORD2VEC] Training a new word2vec model...')
        self.model = W2V(sentences=corpus, vector_size=self.embedding_size, 
                         window=self.context_window, epochs=self.epochs, 
                         workers=cpu_count(), min_count=0, sg=1, negative=5, 
                         sample=0, seed=self.seed)
        print(f'           {self.destination}.word2vec trained')
        print(f'           Total embeddings: {len(self.model.wv.index_to_key)}')
        if save:
            self.model.save(f'{self.destination}.word2vec')
            print(f'           Model {self.destination}.word2vec saved')

    def update(self, corpus, save=False):
        """_summary_

        Parameters
        ----------
        corpus : _type_
            _description_
        save : bool, optional
            _description_, by default False
        """
        print(f'[WORD2VEC] Updating {self.source}.word2vec model ...')
        self.model.build_vocab(corpus, update=True, trim_rule=None)
        self.model.train(corpus, total_examples=self.model.corpus_count, 
                         epochs=self.epochs)
        print(f'           {self.destination}.word2vec trained')
        print(f'           Total embeddings: {len(self.model.wv.index_to_key)}')
        if save:
            self.model.save(f'{self.destination}.word2vec') 
            print(f'           Model {self.destination}.word2vec saved')

    def get_embeddings(self, ips=None, labels=None, dst_path=None):
        """_summary_

        Parameters
        ----------
        ips : list, optional
            list of senders for which retrieve the embeddings. If None get the
            full embeddings matrix stored in the model, by default None
        labels : _type_, optional
            _description_, by default None
        dst_path : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # Return a dataframe with the provided IPs as index
        # If provided also labels they are set as last column named `class`
        # Labels must be provided as another pandas dataframe with a column
        # named `ip` and a column named `class`
        if type(ips)==type(None):
            ips = [x for x in self.model.wv.index_to_key]
        embeddings = self.model.wv.vectors 
        embeddings = pd.DataFrame(embeddings, index=ips)
        
        if type(labels)==pd.core.frame.DataFrame:
            embeddings = embeddings.reset_index()\
                                .rename(columns={'index':'ip'})\
                                .merge(labels, on='ip', how='left')\
                                .set_index('ip').fillna('unknown')
        elif type(labels) == list:
            if type(ips)==type(None):
                raise ValueError(f'Providing labels requires also ips')
            elif len(labels)!=len(ips):
                raise ValueError(f'Length mismatch:')

        if type(dst_path)!=type(None):
            embeddings.to_csv(f'{dst_path}.csv.gz')
            
        return embeddings

    def del_embeddings(self, to_drop, dst_path=None):
        """_summary_

        Parameters
        ----------
        to_drop : _type_
            _description_
        dst_path : _type_, optional
            _description_, by default None
        """
        idx = np.isin(self.model.wv.index2word, to_drop)
        idx = np.where(idx==True)[0]
        self.model.wv.index2word = list(
                              np.delete(self.model.wv.index2word, idx, axis=0))
        self.model.wv.vectors = np.delete(self.model.wv.vectors, idx, axis=0)
        self.model.trainables.syn1neg = np.delete(
                                            self.model.trainables.syn1neg, idx, axis=0)
        list(map(self.model.wv.vocab.__delitem__, 
                 filter(self.model.wv.vocab.__contains__,to_drop)))

        for i, word in enumerate(self.model.wv.index2word):
            self.model.wv.vocab[word].index = i

        if type(dst_path)!=type(None):
            self.model.save(f'{dst_path}.word2vec')