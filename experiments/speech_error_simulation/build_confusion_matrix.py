import argparse

from common_utils.multiwoz_data import MultiWOZData
from experiments.noisy_speech_data.core import NOISE_LIST
from experiments.speech_error_simulation.error_simulator import SpeechErrorSimulator

def main(part, side, noise_type):
    simulator = SpeechErrorSimulator.from_transcript_data(part, side, noise_type)
    simulator.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=MultiWOZData.parts, default="train")
    parser.add_argument("--side", choices=MultiWOZData.sides, default="sys")
    parser.add_argument("--noise_type", type=str, default="background(0)")
    args = parser.parse_args()
    main(part=args.part, side=args.side, noise_type=args.noise_type)