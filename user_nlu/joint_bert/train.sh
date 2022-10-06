#!/bin/bash

train_script_path="/home/ohashi/dev/antor/ConvLab-2/convlab2/nlu/jointBERT"

current_dpath=$(cd $(dirname $0);pwd)
config_path="${current_dpath}/$1"

cp ${config_path} ${train_script_path}/multiwoz/configs # config fileをconvlab2の学習スクリプトのところにコピーする
python ${train_script_path}/train.py --config_path ${config_path} # 学習実行