# Correct darknet traffic start from 2021/03
START  = '202103'
# Global path of the darknet traces
TRACES  = f'/share/smartdata/security/darknet/logs/it_v4/trace-{START}'
DATA = '/share/smartdata/huawei/darknet_graph/coNEXT/'

CORPUS  = f'{DATA}/corpus'
GROUNDTRUTH = f'{DATA}/groundtruth/ground_truth_full.csv.gz'
MODELS = f'{DATA}/models'
DATASETS = f'{DATA}/datasets'
GRIDSEARCH = f'{DATA}/gridsearch'
PROTO_CONVERSION = {'6':'tcp', '1':'icmp', '17':'udp', '47':'gre'}

import os
traces = os.listdir(TRACES[:-12])
traces_path = ''
for i in range(2, 32):
    if i<10:
        traces_path+='file://'+TRACES[:-12]+f'trace-2021030{i}*/packets.log.gz,'
    else:
        traces_path+='file://'+TRACES[:-12]+f'trace-202103{i}*/packets.log.gz,'

DEBUG = traces_path[:-1]