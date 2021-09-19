# Correct darknet traffic start from 2021/03
START  = '202103'
# Global path of the raw darknet traces
TRACES  = f'raw/trace-{START}'
# Global path of the data folder
DATA = 'coNEXT/'

# Other useful paths
CORPUS  = f'{DATA}/corpus'
GROUNDTRUTH = f'{DATA}/groundtruth/ground_truth_full.csv.gz'
MODELS = f'{DATA}/models'
DATASETS = f'{DATA}/datasets'
GRIDSEARCH = f'{DATA}/gridsearch'

# Mapping between IP protocol number and string identifier
PROTO_CONVERSION = {'6':'tcp', '1':'icmp', '17':'udp', '47':'gre'}

# Manage all the traces for 2021/03
import os
traces = os.listdir(TRACES[:-12])
traces_path = ''
for i in range(2, 32):
    if i<10:
        traces_path+='file://'+TRACES[:-12]+f'trace-{START}0{i}*/packets.log.gz,'
    else:
        traces_path+='file://'+TRACES[:-12]+f'trace-{START}{i}*/packets.log.gz,'

DEBUG = traces_path[:-1]
