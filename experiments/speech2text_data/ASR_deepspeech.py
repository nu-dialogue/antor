"""
Based on `ConvLab-2/convlab2/laug/Speech_Recognition/ASR.py`
"""
from __future__ import absolute_import, division, print_function
import os
import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import json

from deepspeech import Model, version
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


class wav2text():
    def __init__(self, model='models/deepspeech-0.9.3-models.pbmm', scorer='models/deepspeech-0.9.3-models.scorer',
                 audio=None, beam_width=None, lm_alpha=None, lm_beta=None, version=None,
                 extended=False, json=False, candidate_transcripts=3, hot_words=None):
        """
        Original argments of `ConvLab-2/convlab2/laug/Speech_Recognition/ASR.py`
            parser.add_argument('--model', required=False,default='deepspeech-0.9.3-models.pbmm',
                                    help='Path to the model (protocol buffer binary file)')
            parser.add_argument('--scorer', required=False,default='deepspeech-0.9.3-models.scorer',
                                    help='Path to the external scorer file')
            parser.add_argument('--audio', required=False,
                                    help='Path to the audio file to run (WAV format)')
            parser.add_argument('--beam_width', type=int,
                                    help='Beam width for the CTC decoder')
            parser.add_argument('--lm_alpha', type=float,
                                    help='Language model weight (lm_alpha). If not specified, use default from the scorer package.')
            parser.add_argument('--lm_beta', type=float,
                                    help='Word insertion bonus (lm_beta). If not specified, use default from the scorer package.')
            parser.add_argument('--version', action=VersionAction,
                                    help='Print version and exits')
            parser.add_argument('--extended', required=False, action='store_true',
                                    help='Output string from extended metadata')
            parser.add_argument('--json', required=False, action='store_true',
                                    help='Output json from metadata with timestamp of each word')
            parser.add_argument('--candidate_transcripts', type=int, default=3,
                                    help='Number of candidate transcripts to include in JSON output')
            parser.add_argument('--hot_words', type=str,
                                    help='Hot-words and their boosts.')
        """
        self.model = model
        self.scorer = scorer
        self.audio = audio
        self.beam_width = beam_width
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.version = version
        self.extended = extended
        self.json = json
        self.candidate_transcripts = candidate_transcripts
        self.hot_words = hot_words

        print('Loading model from file {}'.format(self.model), file=sys.stderr)
        model_load_start = timer()
        # sphinx-doc: python_ref_model_start
        model_path=os.path.dirname(os.path.abspath(__file__))

        ds = Model(os.path.join(model_path,self.model))
        # sphinx-doc: python_ref_model_stop
        model_load_end = timer() - model_load_start
        print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

        if self.beam_width:
            ds.setBeamWidth(self.beam_width)

        self.desired_sample_rate = ds.sampleRate()



        if self.scorer:
            print('Loading scorer from files {}'.format(self.scorer), file=sys.stderr)
            scorer_load_start = timer()
            ds.enableExternalScorer(os.path.join(model_path,self.scorer))
            scorer_load_end = timer() - scorer_load_start
            print('Loaded scorer in {:.3}s.'.format(scorer_load_end), file=sys.stderr)

            if self.lm_alpha and self.lm_beta:
                ds.setScorerAlphaBeta(self.lm_alpha, self.lm_beta)

        if self.hot_words:
            print('Adding hot-words', file=sys.stderr)
            for word_boost in self.hot_words.split(','):
                word,boost = word_boost.split(':')
                ds.addHotWord(word,float(boost))
        self.ds=ds
        
    def run(self,audio):
        fin = wave.open(audio, 'rb')
        fs_orig = fin.getframerate()
        if fs_orig != self.desired_sample_rate:
            print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, self.desired_sample_rate), file=sys.stderr)
            fs_new, audio = convert_samplerate(self.audio, self.desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

        audio_length = fin.getnframes() * (1/fs_orig)
        fin.close()

        inference_start = timer()
        # sphinx-doc: python_ref_inference_start
        text=self.ds.stt(audio)
        #print(text)
        # sphinx-doc: python_ref_inference_stop
        inference_end = timer() - inference_start
        #print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
        return text