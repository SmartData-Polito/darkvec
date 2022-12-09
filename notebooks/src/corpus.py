import numpy as np
from pandas import DatetimeIndex
from config import LANGUAGES 

def drop_duplicates(corpus):
    new_corpus = []
    for service in corpus:
        _prev = np.array(service)
        _next = np.roll(_prev, -1)
        _next[-1] = 'NULL'
        service = _prev[_prev!=_next]
        new_corpus.append(list(service))
    
    return new_corpus


def get_top_ports(dev, TOP):
    try: dev = dev.drop(columns=['serv'])
    except: pass
    topports = dev.value_counts('pp')
    top_p = topports.iloc[:TOP].index
    temp__ = dev.drop(columns=['ts']).reset_index()
    idx = temp__[temp__.pp.isin(top_p)].index
    temp__.loc[idx,'serv'] = temp__.loc[idx,'pp']
    temp__ = temp__.fillna('other')
    temp__.index = DatetimeIndex(temp__.ts)
    
    return temp__

def get_services(x):
    if x in LANGUAGES: 
        return LANGUAGES[x]
    else: 
        x = x.split('/')[0]
        if x!='-':
            if int(x) >= 0 and int(x) <= 1023: 
                return 'unk_sys'
            elif int(x) >= 1024 and int(x) <= 49151: 
                return 'unk_usr'
            elif int(x) >= 49152 and int(x) <= 65535: 
                return 'unk_eph'
        else: 
            return 'icmp'

def get_hours(x):
    hh = x.hour
    if hh < 10: hh = f'0{hh}'
    dd = x.day
    if dd < 10: dd = f'0{dd}'
    mm = x.month
    if mm < 10: mm = f'0{mm}'
    yy = x.year
    if yy < 10: yy = f'0{yy}'
    
    return f'{yy}_{mm}_{dd}_{hh}'


def get_corpus(data, without_duplicates=True, services='auto', top_ports=None):
    # Define 1h sequences
    data['hour'] = data.ts.apply(get_hours)
    
    if services=='single':
        rows = data.groupby(['hour']).agg({'ip':list})\
                   .sort_values(['hour']).values
        corpus = [x[0] for x in rows]

    elif services=='auto':
        if not isinstance(top_ports, int):
            raise Exception('top_ports parameter missing. Provide the number '\
                            'of top ports to use as services')
        data = get_top_ports(data, top_ports)
        rows = data.groupby(['serv', 'hour']).agg({'ip':list})\
                   .sort_values(['hour', 'serv']).values
        corpus = [x[0] for x in rows]

    elif services=='hybrid':
        if not isinstance(top_ports, int):
            raise Exception('top_ports parameter missing. Provide the number '\
                            'of top ports to use as services')
        data = get_top_ports(data, top_ports)
        rows1 = data.groupby(['serv', 'hour']).agg({'ip':list})\
                .sort_values(['hour', 'serv']).values
        corpus1 = [x[0] for x in rows1]
        
        data['serv'] = data.pp.apply(get_services)
        rows2 = data.groupby(['serv', 'hour']).agg({'ip':list})\
                .sort_values(['hour', 'serv']).values
        corpus2 = [x[0] for x in rows2]
        corpus = corpus1 + corpus2

    elif services=='dks':
        data['serv'] = data.pp.apply(get_services)
        rows = data.groupby(['serv', 'hour']).agg({'ip':list})\
                   .sort_values(['hour', 'serv']).values
        corpus = [x[0] for x in rows]

    if without_duplicates:
        corpus = drop_duplicates(corpus)

    return corpus