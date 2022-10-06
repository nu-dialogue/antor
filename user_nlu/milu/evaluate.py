"""
Based on `ConvLab-2/convlab2/nlu/milu/evaluate.py`
"""

import os
from os.path import dirname, abspath
import sys
import argparse

CURRENT_DPATH = dirname(abspath(__file__))
ROOT_DPATH = dirname(dirname(dirname(CURRENT_DPATH)))
sys.path.append(ROOT_DPATH)

from convlab2.nlu.milu.evaluate import evaluate_from_args
from simulator.nlu.milu import dataset_reader

parser = argparse.ArgumentParser(description="Evaluate the specified model + dataset.")
parser.add_argument('archive_relative_dpath', type=str, help='path to an archived trained model')

parser.add_argument('--input_file', type=str,
                        default=os.path.join(ROOT_DPATH, "ConvLab-2/data/multiwoz/test.json.zip"),
                        help='path to the file containing the evaluation data')

parser.add_argument('--output-file', type=str, help='path to output file')

parser.add_argument('--weights-file',
                        type=str,
                        help='a path that overrides which weights file to use')

cuda_device = parser.add_mutually_exclusive_group(required=False)
cuda_device.add_argument('--cuda-device',
                            type=int,
                            default=-1,
                            help='id of GPU to use (if any)')

parser.add_argument('-o', '--overrides',
                        type=str,
                        default="",
                        help='a JSON structure used to override the experiment configuration')

parser.add_argument('--batch-weight-key',
                        type=str,
                        default="",
                        help='If non-empty, name of metric used to weight the loss on a per-batch basis.')

parser.add_argument('--extend-vocab',
                        action='store_true',
                        default=False,
                        help='if specified, we will use the instances in your new dataset to '
                            'extend your vocabulary. If pretrained-file was used to initialize '
                            'embedding layers, you may also need to pass --embedding-sources-mapping.')

parser.add_argument('--embedding-sources-mapping',
                        type=str,
                        default="",
                        help='a JSON dict defining mapping from embedding module path to embedding'
                        'pretrained-file used during training. If not passed, and embedding needs to be '
                        'extended, we will try to use the original file paths used during training. If '
                        'they are not available we will use random vectors for embedding extension.')

if __name__ == "__main__":
    args = parser.parse_args()
    args.archive_file = os.path.join(CURRENT_DPATH, args.archive_relative_dpath, "model.tar.gz")
    evaluate_from_args(args)