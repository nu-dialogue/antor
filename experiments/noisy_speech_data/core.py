import os
import re
import shutil
import functools
import soundfile
from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations import (
    AddBackgroundNoise,
    ApplyImpulseResponse,
    RoomSimulator,
    AddGaussianNoise,
    AddGaussianSNR,
    HighPassFilter
)
from common_utils.path import ROOT_DPATH

NOISE_LIST = [
    "none",
    "background",
    "impulse_response",
    "room_simulator",
    "gaussian",
    "gaussian_snr",
    "high_pass_filter"
]

def _none(src_audio_fpath, output_fpath):
    shutil.copy2(src_audio_fpath, output_fpath)

def _augment(augmentor, src_audio_fpath, output_fpath):
    samples, sample_rate = load_sound_file(file_path=src_audio_fpath, sample_rate=None)
    augmented_samples = augmentor(samples=samples, sample_rate=sample_rate)
    soundfile.write(output_fpath, data=augmented_samples, samplerate=sample_rate, format='WAV')

def get_noise_func(noise_type):
    noise_name, *params = re.split(r'[\(\)]', noise_type)
    if params:
        params = params[0].split(",")
    assert noise_name in NOISE_LIST

    if noise_name == "none":
        return _none

    elif noise_name == "background":
        background_noise_fpath = os.path.join(ROOT_DPATH,
                                              "experiments/noisy_speech_data/datasets",
                                              f"background_noise/audio")
        kwargs = {"sounds_path": background_noise_fpath, "p": 1.0}
        if params:
            snr_in_db = float(params[0])
            kwargs.update({"min_snr_in_db": snr_in_db, "max_snr_in_db": snr_in_db})
        augmentor = AddBackgroundNoise(**kwargs)

    elif noise_name == "impulse_response":
        ir_dpath = os.path.join(ROOT_DPATH,
                                "experiments/noisy_speech_data/datasets",
                                "impulse_responses/audio")
        kwargs = {"ir_path": ir_dpath, "p": 1.0}
        augmentor = ApplyImpulseResponse(**kwargs)

    elif noise_name == "room_simulator":
        kwargs = {"p": 1.0}
        augmentor = RoomSimulator(**kwargs)

    elif noise_name == "gaussian":
        kwargs = {"p": 1.0}
        augmentor = AddGaussianNoise(**kwargs)

    elif noise_name == "gaussian_snr":
        kwargs = {"p": 1.0}
        augmentor = AddGaussianSNR(**kwargs)

    elif noise_name == "high_pass_filter":
        kwargs = {"p": 1.0}
        augmentor = HighPassFilter(**kwargs)
        
    else:
        raise NotImplementedError
    
    return functools.partial(_augment, augmentor=augmentor)
