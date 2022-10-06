import os
import argparse
from distutils.util import strtobool
from tqdm import tqdm
from logging import getLogger

from common_utils.path import ROOT_DPATH
from experiments.text2speech_data.TTS import text2wav
from common_utils.multiwoz_data import MultiWOZData
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

def main(part, side, rm_ws_before_punc, gcp_secret_key_fpath):
    multiwoz_data = MultiWOZData(rm_ws_before_punc=rm_ws_before_punc)
    tts = text2wav(secret_key_fpath=gcp_secret_key_fpath)
    outputs_dpath = os.path.join(ROOT_DPATH,
                                 "experiments/text2speech_data/outputs/multiwoz",
                                 part)
    pbar = tqdm(multiwoz_data[part])
    for dial_name in pbar:
        pbar.set_description(f"Processing {dial_name}")
        dial_dpath = os.path.join(outputs_dpath, dial_name)
        os.makedirs(dial_dpath, exist_ok=True)
        for i_turn, side_turn, turn in multiwoz_data.iter_dialog_log(part=part, dial_name=dial_name):
            if side_turn != side:
                continue
            output_fname = os.path.join(dial_dpath, str(i_turn))
            tts.run(text=turn["text"], output_fname=output_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=MultiWOZData.parts, default="train")
    parser.add_argument('--side', choices=MultiWOZData.sides, default="sys")
    parser.add_argument('--rm_ws_before_punc', type=strtobool, default=True)
    parser.add_argument("--gcp_secret_key_fpath", type=str, default="./secret_key.json")
    args = parser.parse_args()
    main(part=args.part, side=args.side, rm_ws_before_punc=args.rm_ws_before_punc,
         gcp_secret_key_fpath=args.gcp_secret_key_fpath)