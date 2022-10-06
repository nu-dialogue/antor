import os
import argparse
from tqdm import tqdm
from logging import getLogger

from common_utils.path import ROOT_DPATH
from experiments.noisy_speech_data.core import get_noise_func
from common_utils.multiwoz_data import MultiWOZData
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

def main(part, side, noise_type):
    src_audio_dpath = os.path.join(ROOT_DPATH, "experiments/text2speech_data/outputs/multiwoz", part)
    output_dpath = os.path.join(ROOT_DPATH, "experiments/noisy_speech_data/outputs/multiwoz",
                                noise_type, part)
    multiwoz_data = MultiWOZData()
    apply_noise = get_noise_func(noise_type=noise_type)
    pbar = tqdm(list(multiwoz_data[part]))
    for dial_name in pbar:
        pbar.set_description(f"Processing {dial_name}")
        for i, s, turn in multiwoz_data.iter_dialog_log(part, dial_name):
            src_audio_fpath = os.path.join(src_audio_dpath, dial_name, f'{i}.wav')
            if s != side or not os.path.isfile(src_audio_fpath):
                continue
            output_fpath = os.path.join(output_dpath, dial_name, f'{i}.wav')
            os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
            apply_noise(src_audio_fpath=src_audio_fpath, output_fpath=output_fpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=MultiWOZData.parts, default="train")
    parser.add_argument("--side", choices=MultiWOZData.sides, default="sys")
    parser.add_argument("--noise_type", type=str, default="background(0)")
    args = parser.parse_args()
    main(part=args.part, side=args.side, noise_type=args.noise_type)