import json
import os
import argparse
from tqdm import tqdm
from logging import getLogger

from common_utils.path import ROOT_DPATH
from experiments.speech2text_data.ASR_google import wav2text
from common_utils.multiwoz_data import MultiWOZData
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

def main(part, side, noise_type, gcp_secret_key_fpath):
    audio_dpath = os.path.join(ROOT_DPATH,
                               "experiments/noisy_speech_data/outputs/multiwoz",
                               noise_type, part)
    asr = wav2text(secret_key_fpath=gcp_secret_key_fpath)
    multiwoz_data = MultiWOZData(rm_ws_before_punc=True)

    output_data = {}
    output_fpath = os.path.join(ROOT_DPATH, "experiments/speech2text_data/outputs/multiwoz",
                                noise_type, f"{part}.json")
    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
    pbar = tqdm(list(multiwoz_data[part]))
    for dial_name in pbar:
        pbar.set_description(f"Processing {dial_name}")
        output_data[dial_name] = {"log": []}
        for i, s, turn in multiwoz_data.iter_dialog_log(part, dial_name):
            audio_fpath = os.path.join(audio_dpath, dial_name, f"{i}.wav")
            if s != side or not os.path.isfile(audio_fpath):
                output_data[dial_name]["log"].append({})
            else:
                response = asr.run(audio_fpath)
                if response.results:
                    result = {
                        "src_text": turn["text"],
                        "transcript": response.results[0].alternatives[0].transcript,
                        "confidence": response.results[0].alternatives[0].confidence,
                        "time": response.results[0].result_end_time.seconds,
                    }
                else:
                    logger.warning(f"Failed to transcribe {audio_fpath}")
                    result = {
                        "src_text": turn["text"],
                        "transcript": "",
                        "confidence": None,
                        "time": None,
                    }
                output_data[dial_name]["log"].append(result)
            # Save a transcript every turn for safety
            json.dump(output_data, open(output_fpath, "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=MultiWOZData.parts, default="train")
    parser.add_argument("--side", choices=MultiWOZData.sides, default="sys")
    parser.add_argument("--noise_type", type=str, default="background(0)")
    parser.add_argument("--gcp_secret_key_fpath", type=str, default="./secret_key.json")
    args = parser.parse_args()
    main(part=args.part, side=args.side, noise_type=args.noise_type,
         gcp_secret_key_fpath=args.gcp_secret_key_fpath)