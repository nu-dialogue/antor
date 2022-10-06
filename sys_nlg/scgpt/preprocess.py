# -*- coding: utf-8 -*-
"""
Based on ConvLab-2/convlab2/nlg/scgpt/multiwoz/preprocess.py
"""

import os
import json
from convlab2.nlg.scgpt.utils import dict2dict, dict2seq
import zipfile

def read_zipped_json(filepath, filename):
    print("zip file path = ", filepath)
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))

cur_dir = os.path.dirname(os.path.abspath(__file__)) 
from common_utils.path import ROOT_DPATH
data_dir = os.path.join(ROOT_DPATH, f"ConvLab-2/data/multiwoz")

keys = ['train', 'val', 'test']
data = {}
for key in keys:
    data_key = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
    print('load {}, size {}'.format(key, len(data_key)))
    data = dict(data, **data_key)

with open(os.path.join(data_dir, 'valListFile'), 'r') as f:
    val_list = f.read().splitlines()
with open(os.path.join(data_dir, 'testListFile'), 'r') as f:
    test_list = f.read().splitlines()
    
results = {}
results_val = {}
results_test = {}

for title, sess in data.items():
    logs = sess['log']
    turns = []
    turn = {'turn':0, 'sys':'', 'sys_da':''}
    current_domain = None
    for i, diag in enumerate(logs):
        text = diag['text']
        da = diag['dialog_act']
        span = diag['span_info']
        if i % 2 == 0:
            turn['usr'] = text
            if current_domain:
                da = eval(str(da).replace('Booking', current_domain))
                span = eval(str(span).replace('Booking', current_domain))
            turn['usr_da'] = da
            turn['usr_span'] = span
            turns.append(turn)
        else:
            turn = {'turn': i//2 +1}
            turn['sys'] = text
            turn['sys_da'] = da
            turn['sys_span'] = span
        for key in da:
            domain = key.split('-')[0]
            if domain not in ['general', 'Booking']:
                current_domain = domain
    title = title
    if title in val_list:
        current = results_val
    elif title in test_list:
        current = results_test
    else:
        current = results
    current[title] = turns
    
results = eval(str(results).replace(" n't", " not"))
results_val = eval(str(results_val).replace(" n't", " not"))
results_test = eval(str(results_test).replace(" n't", " not"))

def write_file(name, data):
    with open(f'{name}.txt', 'w', encoding='utf-8') as f:
        for ID in data:
            sess = data[ID]
            for turn in sess:
                if not turn['sys_da']: # use sys_da instead of usr_da
                    continue
                turn['sys_da'] = eval(str(turn['sys_da']).replace('Bus','Train')) # use sys_da instead of usr_da
                da_seq = dict2seq(dict2dict(turn['sys_da'])).replace('&', 'and') # use sys_da instead of usr_da
                da_uttr = turn['sys'].replace(' bus ', ' train ').replace('&', 'and')
                f.write(f'{da_seq} & {da_uttr}\n')

if not os.path.exists(os.path.join(cur_dir,'data')):
    os.makedirs(os.path.join(cur_dir, 'data'))
write_file(os.path.join(cur_dir, 'data/train'), results)
write_file(os.path.join(cur_dir, 'data/val'), results_val)
write_file(os.path.join(cur_dir, 'data/test'), results_test)
