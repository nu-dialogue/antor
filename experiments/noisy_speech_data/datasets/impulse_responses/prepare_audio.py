import io
import os
from tqdm import tqdm
import librosa
import soundfile
import zipfile
import argparse

# https://librosa.org/doc/main/ioformats.html#read-file-like-objects

def main(resample_rate=None):
    source_audio_zippath = "Audio.zip"
    audio_output_dpath = "audio"

    with zipfile.ZipFile(source_audio_zippath) as audio_zip:
        for info in tqdm(audio_zip.infolist()):
            if not info.filename.endswith(".wav"):
                continue
            with audio_zip.open(info.filename) as audio_file:
                filename = os.path.basename(info.filename)
                tmp = io.BytesIO(audio_file.read())
                samples, sample_rate = soundfile.read(tmp)
            if resample_rate is not None:
                samples = librosa.resample(samples,
                                           orig_sr=sample_rate,
                                           target_sr=resample_rate)
                sample_rate = resample_rate
            audio_output_fpath = os.path.join(audio_output_dpath, filename)
            os.makedirs(os.path.dirname(audio_output_fpath), exist_ok=True)
            soundfile.write(audio_output_fpath,
                            data=samples, samplerate=sample_rate, format='WAV')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample_rate", type=int, default=16000)
    args = parser.parse_args()
    main(resample_rate=args.resample_rate)