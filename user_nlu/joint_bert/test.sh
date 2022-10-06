#!/bin/bash

train_script_path="/data/group1/z44383r/dev/ppn-nlg/ConvLab-2/convlab2/nlu/jointBERT"

current_dpath=$(cd $(dirname $0);pwd)
config_path="${current_dpath}/$1"

python ${train_script_path}/test.py --config_path ${config_path} # 学習実行