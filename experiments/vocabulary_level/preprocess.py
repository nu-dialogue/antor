import argparse
import os
import pandas as pd
import json
import re
from tqdm import tqdm
from collections import Counter, defaultdict
import spacy
from common_utils.multiwoz_data import make_tags, flatten_da, MultiWOZData
from common_utils.path import ROOT_DPATH

nlp = spacy.load('en_core_web_sm')

def compute_vocab_level_by_turn(data, vocabularies):
    vocabulary_level_per_turn = defaultdict(list)
    for dialog_id, dialog in tqdm(data.items()):
        for turn in dialog["log"]:
            vocab_level = 'NonAlpha+Stop'
            doc = nlp(turn["text"])
            tags = make_tags(length=len(doc), span_info=turn["span_info"])
            da = flatten_da(turn["dialog_act"])

            for token, tag in zip(doc, tags):
                if not token.is_alpha: # アルファベットではない場合は除外
                    continue
                elif token.is_stop: # stopwordは除外
                    continue
                # elif tag != "O": # slot情報が含まれている場合は除外
                #     continue
                elif token.lemma_.lower() in vocabularies['A1']:
                    if vocab_level in ['NonAlpha+Stop']:
                        vocab_level = 'A1'
                    continue
                elif token.lemma_.lower() in vocabularies['A2']:
                    if vocab_level in ['NonAlpha+Stop', 'A1']:
                        vocab_level = 'A2'
                    continue
                elif token.lemma_.lower() in vocabularies['B1']:
                    if vocab_level in ['NonAlpha+Stop', 'A1', 'A2']:
                        vocab_level = 'B1'
                    continue
                elif token.lemma_.lower() in vocabularies['B2']:
                    if vocab_level in ['NonAlpha+Stop', 'A1', 'A2', 'B1']:
                        vocab_level = 'B2'
                    continue
                else:
                    vocab_level = 'OOV'
                    break
                
            vocabulary_level_per_turn[dialog_id].append(vocab_level)
    return vocabulary_level_per_turn

def print_level_freqs(vocab_level):
    level_freq = []
    for turns in vocab_level.values():
        level_freq += turns
    level_freq = Counter(level_freq)
    print(level_freq)

def main(cefr_fpath):
        
    multiwoz_data = MultiWOZData(rm_ws_before_punc=True)
    df = pd.read_csv(cefr_fpath)

    levels = {
        'A1': df[df['CEFR']=='A1'],
        'A2': df[(df['CEFR']=='A2')],
        'B1': df[(df['CEFR']=='B1')],
        'B2': df[(df['CEFR']=='B2')],
    }

    vocabularies = defaultdict(list)
    for level in levels:
        vocabulary = []
        for word in levels[level]['headword']:
            word = word.lower()
            vocabulary += word.split("/")
        vocabularies[level] = set(vocabulary)

    for part in ["val", "test", "train"]:
        print(f"\n\nComputing {part}'s level...")
        vocab_level = compute_vocab_level_by_turn(data=multiwoz_data[part],
                                                vocabularies=vocabularies)

        vocab_level_dpath = os.path.join(ROOT_DPATH, "experiments/vocabulary_level/outputs/cefrj")
        os.makedirs(vocab_level_dpath, exist_ok=True)
        vocab_level_fpath = os.path.join(vocab_level_dpath, f"{part}.json")
        json.dump(vocab_level, open(vocab_level_fpath, "w"), indent=4)
        print_level_freqs(vocab_level)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cefr_wordlist_fpath", type=str, default="./olp-en-cefrj/cefrj-vocabulary-profile-1.5.csv")
    args = parser.parse_args()
    
    main(cefr_fpath=args.cefr_wordlist_fpath)