from config import *
from src.word2vec import Word2Vec
from src.knnClassifier import KnnClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

###############################################################################
# Jupyter-notebooks 01-darknet-overview.ipynb
###############################################################################
def get_ips_ecdf(dnet):
    """Get the cumulative sum of the distinct IPs per day seen over 30 days of 
    darknet traffic
    Parameters
    ----------
    dnet : pandas.DataFrame
        distinc IPs per day
    Returns
    -------
    pandas.DataFrame
        cumulative sum of IPs per day
    """
    prev = dnet.iloc[0]['ip']
    day = dnet.index[0]
    cumuls = [(day, len(prev))]
    for i in range(1, dnet.shape[0]):
        day = dnet.index[i]
        new = dnet.iloc[i]['ip'].union(prev)
        cumuls.append((day, len(new)))
        prev = new
        
    return cumuls

def get_ip_set_by_day(dnet):
    """Get the number of distinc IPs per day
    Parameters
    ----------
    dnet : pandas.DataFrame
        monthly darknet traffic
    Returns
    -------
    pandas.DataFrame
        distinc IPs per day
    """
    dnet.index = pd.DatetimeIndex(dnet.ts)
    dnet = dnet.resample('D').agg({'ip':set})
    dnet.loc['2021-03-02', 'ip'] = dnet.iloc[1]['ip'].union(dnet.iloc[0]['ip'])
    dnet = dnet.iloc[1:]
    
    return dnet

def get_last_day_stats(df, gt_class):
    """Get the number of senders, packets, ports, and the top-5 ports for the 
    provided ground truth class 
    Parameters
    ----------
    df : pandas.DataFrame
        daily grund truth dataframe
    gt_class : str
        ground truth class to analyze
    Returns
    -------
    tuple
        ground truth class, number of senders, number of packets, number of 
        ports and top-5 ports
    """
    senders   = df[df['class'] == gt_class].ip.unique().shape[0]
    ports     = df[df['class'] == gt_class].pp.unique().shape[0]
    packets   = df[df['class'] == gt_class].shape[0]
    top5ports = df[df['class'] == gt_class].value_counts('pp')[:5]
    
    top5 = ''
    for top_p in top5ports.index:
        port_traffic = df[(df['class'] == gt_class) & (df.pp == top_p)]
        port_traffic_perc = port_traffic.shape[0]*100/packets
        top5+=f'{top_p}({round(port_traffic_perc, 1)}%), '
    top5 = top5[:-2]
    
    return (gt_class, senders, packets, ports, top5)

###############################################################################
# Jupyter-notebooks 03-gridsearch.ipynb
###############################################################################

def get_cols(report):
    return {'Avg. Precision' : round(report['weighted avg']['precision'],2),
            'Avg. Recall' : round(report['weighted avg']['recall'],2),
            'Macro Precision' : round(report['macro avg']['precision'],2),
            'Macro Recall' : round(report['macro avg']['recall'],2),
            'F-Score Mirai-like' : round(report['mirai']['f1-score'],2),
            'F-Score Censys' : round(report['censys']['f1-score'],2),
            'Stretchoid' : round(report['stretchoid']['f1-score'],2),
            'Internet-census' : round(report['internet-census']['f1-score'],2),
            'Binaryedge' : round(report['binaryedge']['f1-score'],2),
            'Sharashka' : round(report['sharashka']['f1-score'],2),
            'Ipip' : round(report['ipip']['f1-score'],2),
            'Shodan' : round(report['shodan']['f1-score'],2),
            'Engin-umich' : round(report['engin-umich']['f1-score'],2)}

def get_knn_stats(ips, lookup):
    print('Retrieving 5-NN statistics')
    columns = [('Single training', 'With duplicates'),
               ('Single training', 'Without duplicates'),
               ('Incremental training', 'Without duplicates')]
    METHODS = ['single', 'single', 'incremental']
    DUPLICATES = ['with', 'without', 'without']
    SERVS = {'dks':'DKS', 'auto':'AS', 'hybrid':'HS'}
    TABLE = dict()
    for i in range(3):
        for service in ['dks', 'auto', 'hybrid']:
            cname = (columns[i][0], columns[i][1], SERVS[service])
            method = METHODS[i]
            duplicates = DUPLICATES[i]

            model_path_name = f'gridsearch_corpus_{service}_{method}_{duplicates}'
            print(f'Loading {model_path_name}')
            w2v = Word2Vec(mname=f'{MODELS}/gridsearch/{model_path_name}')
            w2v.load_model()
            embeddings = w2v.get_embeddings(ips, lookup)
            print('Running 5-NN classifier')
            knn = KnnClassifier(embeddings, 5)
            knn.fit_predict()
            knnlabels = lookup[lookup['class']!='unknown']['class'].unique()
            report = knn.get_report(output_dict=True, labels=knnlabels)
            TABLE[cname] = get_cols(report)

    tab = pd.DataFrame(TABLE)
    return tab

def get_corpus_stats():
    print('Retrieving corpus statistics')

    servs = {'dks':'DKS', 'auto':'AS', 'hybrid':'HS'}
    dupls = {'with':'With duplicates', 'without':'Without duplicates'}
    meths = {'single':'Single training', 'incremental':'Incremental training'}
    rows  = {'ndocs':'# Documents', 'avg_words':'Avg. Doc length', 
             'max_words':'Max. Doc length', 'avg_raw_words':'Avg. Raw doc length', 
             'max_raw_words':'Max. Raw doc length'}
    with open(f'{DATASETS}/corpus_stats.json', 'r') as file:
        corpus_stats = json.loads(file.read())

    temp = dict()
    for k,v in corpus_stats.items():
        if 'gridsearch_corpus' in k:
            mname = '_'.join(k.split('_')[:-1])
            _, _, service, method, duplicates = mname.split('_')
            mname = (meths[method], dupls[duplicates], servs[service])
            if mname in temp:
                for k_, v_ in v.items():
                    temp[mname][k_] = temp[mname][k_] + [v_]
            else:
                temp[mname] = v
                for k_, v_ in v.items():
                    temp[mname][k_] = [v_]
                    
    df = pd.DataFrame(temp)
    df.loc['ndocs'] = df.loc['ndocs'].apply(lambda x: np.sum(x))
    df.loc['avg_words'] = df.loc['avg_words'].apply(lambda x: int(np.mean(x)))
    df.loc['max_words'] = df.loc['max_words'].apply(lambda x: np.max(x))

    df.loc['avg_raw_words'] = df.loc['avg_raw_words'].apply(lambda x: int(np.mean(x)))
    df.loc['max_raw_words'] = df.loc['max_raw_words'].apply(lambda x: np.max(x))
    df = df.drop(index=['min_words', 'min_raw_words'])
    df.index = [rows[x] for x in df.index]
    
    return df

def get_runtime_stats():
    print('Retrieving runtime statistics')
    to_replace = {'hybrid':'HS', 'auto':'AS', 'dks':'DKS',
                  'with':'With duplicates', 'without':'Without duplicates',
                  'single':'Single training', 'incremental':'Incremental training'}
    with open(f'{DATASETS}/runtimes.json', 'r') as file:
        runtimes = json.loads(file.read())

    r_times = dict()
    for k, v in runtimes.items():
        if 'gridsearch_corpus' in k and 'incremental' not in k:
                _, _, service, method, duplicates = k.split('_')
                r_times[(to_replace[method], to_replace[duplicates],to_replace[service])] = v
        elif 'gridsearch_corpus' in k and '2021' in k:
            _, _, service, method, duplicates, day = k.split('_')
            key = (to_replace[method], to_replace[duplicates],to_replace[service])
            if key in r_times:
                r_times[key].append(v)
            else:
                r_times[key] = [v]

    new_r = dict()
    for k, v in r_times.items():
        if type(v) == list:
            avg = round(np.mean(v), 1)
            tot = round(np.sum(v), 1)
        else:
            avg = round(v, 1)
            tot = '>10 000'
        new_r[k] = {'Runtime 1 day [s]':avg,
                    'Runtime 30 days [s]':tot}
    r_tab = pd.DataFrame(new_r)

    return r_tab


def get_heatmap_df(service, ips, lookup):
    hm = dict()
    for c in [5, 25, 50, 75]:
        for e in [52, 100, 152, 200]:
            model_path_name = f'model_gs_{service}_c{c}_e{e}'
            print(f'Loading {model_path_name}')
            w2v = Word2Vec(mname=f'{MODELS}/gridsearch/{model_path_name}')
            w2v.load_model()
            embeddings = w2v.get_embeddings(ips, lookup)

            knn = KnnClassifier(embeddings, 5)
            knn.fit_predict()
            knnlabels = lookup[lookup['class']!='unknown']['class'].unique()
            report = knn.get_report(output_dict=True, labels=knnlabels)
            if c in hm:
                hm[c][e] = report['macro avg']['f1-score']
            else:
                hm[c] = {e:report['macro avg']['f1-score']}
    
    return hm

def extract_gridsearch_runtimes(service):
    with open(f'{DATASETS}/runtimes.json', 'r') as file:
        runs = json.loads(file.read())

    xxx = pd.DataFrame([(k,v) for k,v in runs.items()], columns=['macro', 'val'])
    dk = []
    for c in [5, 25, 50, 75]:
        for e in [52, 100, 152, 200]:
            model_path_name = f'model_gs_{service}_c{c}_e{e}'
            dk.append(model_path_name)

    xxx['sx'] = xxx.macro.apply(lambda x: '_'.join(x.split('_')[:-1]))
    xx = xxx[xxx.sx.isin(dk)].groupby('sx').agg({'val':sum}).reset_index()
    xx['C'] = xx.sx.apply(lambda x: int(x.split('_')[-2].replace('c', '')))
    xx['E'] = xx.sx.apply(lambda x: int(x.split('_')[-1].replace('e', '')))
    xx['val'] = xx['val']/60
    xxx = xx.pivot('C', 'E', 'val')

    
    return xxx


###############################################################################
# Jupyter-notebooks 04-clustering.ipynb
###############################################################################
def get_shs_df(G, df):
    if type(G) == dict:
        pred = [(k, v) for k,v in G['clusters'].items()]
    else:
        pred = [(k, v) for k,v in G.clusters.items()]
    embeddings_df = pd.DataFrame(pred, columns=['ip', 'C'])\
                      .merge(df.reset_index()\
                               .drop(columns=['class']))\
                      .set_index('ip')
    silhouettes = silhouette_samples(X=embeddings_df.drop(columns=['C']),
                                     labels = embeddings_df.C.values, metric='cosine')
    sh_df = pd.DataFrame(silhouettes, columns=['sh'], index=embeddings_df.index)
    sh_df['C'] = embeddings_df.C.values
    
    return sh_df

def get_sh_clustering(cluster, embeddings):
    sh_df = get_shs_df(cluster, embeddings)

    temp = sh_df.reset_index().groupby('C')\
                .agg({'ip':lambda x: len(set(x))})\
                .reset_index().merge(sh_df.groupby('C')\
                                          .agg({'sh':'mean'})\
                .reset_index())
    temp = temp[['ip', 'sh']]
    temp = temp.sort_values('sh')
    temp['coverage'] = np.cumsum(temp.ip)/np.sum(temp.ip)
    
    return temp

###############################################################################
# Jupyter-notebooks 05-clusters-inspection.ipynb
###############################################################################
def get_subnet24(x):
    a, b, c, d = x.split('.')
    
    return f'{a}.{b}.{c}.0'

def get_subnet16(x):
    a, b, c, d = x.split('.')
    
    return f'{a}.{b}.0.0'

def extract_shadowserver_pattern(filtered, sh_df):
    shadowserver = [3, 69, 12, 32]
    dfs = []
    for c in shadowserver:
        c_ip = sh_df[sh_df.C==c].ip
        dfs.append(filtered[filtered.ip.isin(c_ip)])
    cl_traffic = pd.concat(dfs)
    cl_traffic = cl_traffic.merge(sh_df[['ip', 'C']], on='ip', how='left').sort_values(['C', 'ts'])

    ptoken = {v:k for k,v in enumerate(cl_traffic.pp.unique())}
    cl_traffic['ptoken'] = cl_traffic.pp.apply(lambda x: ptoken[x])
    iptoken = {v:k for k,v in enumerate(cl_traffic.ip.unique())}
    cl_traffic['iptoken'] = cl_traffic.ip.apply(lambda x: iptoken[x])

    cl_traffic.index = pd.DatetimeIndex(cl_traffic.ts)
    
    return cl_traffic

def extract_censys_pattern(filtered, sh_df):
    censys = [20, 48, 52, 55, 60]
    dfs = []
    for c in censys:
        c_ip = sh_df[sh_df.C==c].ip
        dfs.append(filtered[filtered.ip.isin(c_ip)])
    cl_traffic = pd.concat(dfs)
    cl_traffic = cl_traffic.merge(sh_df[['ip', 'C']], on='ip', how='left').sort_values(['C', 'ts'])

    ptoken = {v:k for k,v in enumerate(cl_traffic.pp.unique())}
    cl_traffic['ptoken'] = cl_traffic.pp.apply(lambda x: ptoken[x])
    iptoken = {v:k for k,v in enumerate(cl_traffic.ip.unique())}
    cl_traffic['iptoken'] = cl_traffic.ip.apply(lambda x: iptoken[x])

    cl_traffic.index = pd.DatetimeIndex(cl_traffic.ts)
    
    return cl_traffic

