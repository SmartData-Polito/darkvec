from config import *
from src.preprocess import load_raw_data, filter_data, get_next_day
from src.corpus import get_corpus
from src.word2vec import Word2Vec
import numpy as np
import time


def update_runtimes(key, val):
    
    RUNTIMES = f'{DATASETS}/runtimes_new.json'
    try:
        with open(RUNTIMES, 'r') as file:
            runtimes = json.loads(file.read())
    except:
        with open(RUNTIMES.replace('_new', ''), 'r') as file:
            runtimes = json.loads(file.read())
    runtimes[key] = val

    with open(RUNTIMES, 'w') as file:
        file.write(json.dumps(runtimes))


def update_corpus(mname, corpus):
    try:
        with open(f'{DATASETS}/corpus_stats_new.json', 'r') as file:
            corpus_stats = json.loads(file.read())
    except:
        with open(f'{DATASETS}/corpus_stats.json', 'r') as file:
            corpus_stats = json.loads(file.read())

    corpus_stats[mname] = {'ndocs':len(corpus), 
                             'min_words':int(np.min([len(x) for x in corpus])),
                             'avg_words': int(np.mean([len(x) for x in corpus])), 
                             'max_words':int(np.max([len(x) for x in corpus])),
                             'min_raw_words':int(np.min([len(set(x)) for x in corpus])),
                             'avg_raw_words':int(np.mean([len(set(x)) for x in corpus])),
                             'max_raw_words':int(np.max([len(set(x)) for x in corpus]))}
        
    with open(f'{DATASETS}/corpus_stats_new.json', 'w') as file:
        file.write(json.dumps(corpus_stats))

"""        
SAVE = True
gs = [{'corpus':{'services':'hybrid', 'without_duplicates':True, 'top_ports':2500},
       'word2vec':{'c':25, 'e':50, 'epochs':1, 'method':'incremental'}}]

for c in [5, 25, 50, 75]:
    for e in [52, 100, 152, 200]:
        for params in gs:
            params['word2vec']['c'] = c
            params['word2vec']['e'] = e
            mname = f'model_gs_hybrid_c{c}_e{e}'
            
            params['word2vec']['mname'] = mname
            print(params['word2vec']['mname'])
            if params['word2vec']['method'] == 'single':
                DAY = '20210331'
                raw_data = load_raw_data('202103')
            else:
                DAY = '20210302'
                raw_data = load_raw_data(DAY)
            filtered = filter_data(raw_data, DAY)
            corpus = get_corpus(filtered, **params['corpus'])
            
            start_time = time.time()
            model = Word2Vec(**params['word2vec'])
            if params['word2vec']['method'] == 'single':
                model.train(corpus, save=SAVE)
                delta_T = time.time() - start_time
                update_runtimes(mname, delta_T)
            else:
                model.train(corpus, save=False)
                delta_T = time.time() - start_time
                update_runtimes(f'{mname}_{DAY}', delta_T)
                current_day = DAY
                
                while True:
                    current_day = get_next_day(current_day)
                    print(current_day)
                    raw_data = load_raw_data(current_day)
                    filtered = filter_data(raw_data, current_day)
                    corpus = get_corpus(filtered, **params['corpus'])
                    if current_day=='20210331':
                        start_time = time.time()
                        model.update(corpus, save=SAVE)
                        delta_T = time.time() - start_time
                        update_runtimes(f'{mname}_{current_day}', delta_T)
                        break
                    else:
                        start_time = time.time()
                        model.update(corpus, save=False)
                        delta_T = time.time() - start_time
                        update_runtimes(f'{mname}_{current_day}', delta_T)

"""


SAVE = True
"""
gs = [{'corpus':{'services':'auto', 'without_duplicates':True, 'top_ports':2500},
       'word2vec':{'c':25, 'e':50, 'epochs':1, 'method':'incremental'}},
      {'corpus':{'services':'hybrid', 'without_duplicates':True, 'top_ports':2500},
       'word2vec':{'c':25, 'e':50, 'epochs':1, 'method':'incremental'}},
      {'corpus':{'services':'dks', 'without_duplicates':True},
       'word2vec':{'c':25, 'e':50, 'epochs':1, 'method':'incremental'}},
      {'corpus':{'services':'single', 'without_duplicates':True},
       'word2vec':{'c':25, 'e':50, 'epochs':1, 'method':'incremental'}}]
"""

gs = [{'corpus':{'services':'auto', 'without_duplicates':True, 'top_ports':2500},
       'word2vec':{'c':25, 'e':50, 'epochs':1, 'method':'incremental'}},
      {'corpus':{'services':'dks', 'without_duplicates':True},
       'word2vec':{'c':25, 'e':50, 'epochs':1, 'method':'incremental'}},
      {'corpus':{'services':'auto', 'without_duplicates':False, 'top_ports':2500},
       'word2vec':{'c':25, 'e':50, 'epochs':1, 'method':'incremental'}},
      {'corpus':{'services':'dks', 'without_duplicates':False},
       'word2vec':{'c':25, 'e':50, 'epochs':1, 'method':'incremental'}}]


for params in gs:
    mname = f"gs_kofknn_incremental_{params['corpus']['services']}_new_{params['corpus']['without_duplicates']}"

    params['word2vec']['mname'] = mname
    print(params['word2vec']['mname'])
    if params['word2vec']['method'] == 'single':
        DAY = '20210331'
        raw_data = load_raw_data('202103')
    else:
        DAY = '20210302'
        raw_data = load_raw_data(DAY)
    filtered = filter_data(raw_data, DAY)
    corpus = get_corpus(filtered, **params['corpus'])

    start_time = time.time()
    model = Word2Vec(**params['word2vec'])
    if params['word2vec']['method'] == 'single':
        model.train(corpus, save=SAVE)
        delta_T = time.time() - start_time
        update_runtimes(mname, delta_T)
        update_corpus(mname, corpus)
    else:
        model.train(corpus, save=False)
        delta_T = time.time() - start_time
        update_runtimes(f'{mname}_{DAY}', delta_T)
        update_corpus(f'{mname}_{DAY}', corpus)
        current_day = DAY

        while True:
            current_day = get_next_day(current_day)
            print(current_day)
            raw_data = load_raw_data(current_day)
            filtered = filter_data(raw_data, current_day)
            corpus = get_corpus(filtered, **params['corpus'])
            if current_day=='20210331':
                start_time = time.time()
                model.update(corpus, save=SAVE)
                delta_T = time.time() - start_time
                update_runtimes(f'{mname}_{current_day}', delta_T)
                update_corpus(f'{mname}_{current_day}', corpus)
                break
            else:
                start_time = time.time()
                model.update(corpus, save=False)
                delta_T = time.time() - start_time
                update_runtimes(f'{mname}_{current_day}', delta_T)
                update_corpus(f'{mname}_{current_day}', corpus)