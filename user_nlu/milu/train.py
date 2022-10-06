from os.path import dirname, abspath
import sys

ROOT_DPATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DPATH)

from convlab2.nlu.milu.train import argparser, train_model_from_args
from user_nlu.milu import dataset_reader

if __name__ == "__main__":
    args = argparser.parse_args()
    train_model_from_args(args)