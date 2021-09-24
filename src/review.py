import matplotlib.pyplot as plt
import pandas as pd
import json
from config import *

def convert_pp(x, services):       
    try:
        
        x1 = services[x]
    except:
        x1 = unknown_class(x)
    
    return x1

def unknown_class(x):
    x = x.split('/')[0]
    #System Ports
    if x!='-':
        if int(x) >= 0 and int(x) <= 1023: return 'unk_sys'
        # User Ports
        elif int(x) >= 1024 and int(x) <= 49151:return 'unk_usr'
        # Ephemeral Ports
        elif int(x) >= 49152 and int(x) <= 65535:return 'unk_eph'
    else:
        return 'icmp'