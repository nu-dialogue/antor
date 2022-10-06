import os
from tqdm import tqdm
import librosa
import soundfile
import argparse
import os
import pandas as pd

esc50_mata_csv_fpath = "ESC-50/meta/esc50.csv"
esc50_audio_dpath = "ESC-50/audio"
main_data_dpath = "audio"

categories_to_select = [
    "dog",
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects",
    "sheep",
    "crow",

    "rain",
    "sea_waves",
    "crackling_fire",
    "crickets",
    "chirping_birds",
    "water_drops",
    "wind",
    "pouring_water",
    "toilet_flush",
    "thunderstorm",

    "crying_baby",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "footsteps",
    "laughing",
    "brushing_teeth",
    "snoring",
    "drinking_sipping",

    "door_wood_knock",
    "mouse_click",
    "keyboard_typing",
    "door_wood_creaks",
    "can_opening",
    "washing_machine",
    "vacuum_cleaner",
    "clock_alarm",
    "clock_tick",
    "glass_breaking",

    "helicopter",
    "chainsaw",
    "siren",
    "car_horn",
    "engine",
    "train",
    "church_bells",
    "airplane",
    "fireworks",
    "hand_saw"
]

def main(resample_rate=None):
    esc50_meta = pd.read_csv(esc50_mata_csv_fpath)
    esc50_meta = esc50_meta[esc50_meta['category'].isin(categories_to_select)]
    for fname in tqdm(esc50_meta["filename"]):
        audio_fpath = os.path.join(esc50_audio_dpath, fname)
        samples, sample_rate = soundfile.read(audio_fpath)
        if resample_rate is not None:
            samples = librosa.resample(samples,
                                       orig_sr=sample_rate,
                                       target_sr=resample_rate)
            sample_rate = resample_rate
        output_fpath = os.path.join(main_data_dpath, fname)
        os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
        soundfile.write(output_fpath,
                        data=samples, samplerate=sample_rate, format='WAV')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample_rate", type=int, default=16000)
    args = parser.parse_args()
    main(resample_rate=args.resample_rate)