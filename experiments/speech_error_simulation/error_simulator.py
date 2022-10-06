import os
import json
from logging import getLogger
from typing import Tuple
from tqdm import tqdm
import numpy as np
import jiwer
from collections import Counter
import spacy
from common_utils.log import set_logger
from common_utils.multiwoz_data import MultiWOZData
from common_utils.path import ROOT_DPATH
from experiments.speech_error_simulation.utils import get_confusion, tokenize, detokenize, OPETYPE

logger = getLogger(__name__)
set_logger(logger)
nlp = spacy.load('en_core_web_sm')

class SpeechErrorSimulator:
    def __init__(self, word2id, id2word, confusion_matrix, meta_info) -> None:
        self.word2id = word2id
        self.id2word = id2word
        self.confusion_matrix = confusion_matrix
        
        self.meta_info = meta_info
        self.part = meta_info['part']
        self.side = meta_info['side']
        self.noise_type = meta_info['noise_type']
        self.alpha_only = meta_info['alpha_only']
        self.del_word = meta_info['del_word']

    @classmethod
    def get_save_dpath(self, part, side, noise_type):
        save_dpath = os.path.join(ROOT_DPATH,
                                  "experiments/speech_error_simulation/outputs/multiwoz",
                                  noise_type, part, side)
        return save_dpath

    @classmethod
    def from_transcript_data(cls, part, side, noise_type, alpha_only=False, del_word='<del>'):
        transcript_fpath = os.path.join(ROOT_DPATH,
                                        "experiments/speech2text_data/outputs/multiwoz",
                                        noise_type,
                                        f"{part}.json")
        trainscript_data = json.load(open(transcript_fpath))
        multiwoz_data = MultiWOZData()
        wers = []
        equals, substitusions, insertions, deletions = [], [], [], []
        pbar = tqdm(multiwoz_data[part], desc='Computing WER...')
        for dial_name in pbar:
            if dial_name not in trainscript_data:
                logger.warning(f"There is no transcript for {dial_name}")
                break
            pbar.set_description(f'Computing WER... {dial_name}')
            for i, s, _ in multiwoz_data.iter_dialog_log(part=part, dial_name=dial_name):
                try:
                    turn = trainscript_data[dial_name]["log"][i]
                except (IndexError, KeyError):
                    logger.warning(f"There is no transcript from turn {i} in {dial_name}")
                    break
                if s != side or not turn:
                    continue
                wers.append(jiwer.wer(turn["src_text"], turn["transcript"]))
                eq, sub, ins, dlt = get_confusion(turn["src_text"], turn["transcript"])
                equals += eq
                substitusions += sub
                insertions += ins
                deletions += dlt

        equals_with_tgt = [(token, token) for token in equals]
        deletions_with_tgt = [(token, del_word) for token in deletions]
        confusions = equals_with_tgt + substitusions + deletions_with_tgt

        vocabs = []
        pbar = tqdm(Counter(confusions).most_common(), desc='Computing vocab...')
        for token_pair, count in pbar:
            for token in token_pair:
                if token in vocabs:
                    continue
                if alpha_only and not nlp(token)[0].is_alpha:
                    continue
                vocabs.append(token)

        id2word = vocabs
        word2id = {token: i for i, token in enumerate(vocabs)}
        confusion_matrix = np.zeros([len(id2word), len(id2word)], dtype=np.float64)

        # count
        for (src_token, tgt_token), count in Counter(confusions).items():
            try:
                src_id = word2id[src_token]
                tgt_id = word2id[tgt_token]
                confusion_matrix[src_id, tgt_id] = count
            except KeyError:
                continue

        # count to prob
        for i in range(len(id2word)):
            if not confusion_matrix[i].sum():
                # process tokens not having tgt_token
                confusion_matrix[i, i] = 1

        meta_info = {
            "part": part,
            "side": side,
            "noise_type": noise_type,
            "alpha_only": alpha_only,
            "del_word": del_word,
            "wer": float(np.array(wers).mean()),
            "num_turns": len(wers),
            "num_equals": len(equals),
            "num_substitusions": len(substitusions),
            "num_insertions": len(insertions),
            "num_deletions": len(deletions),
        }
        return cls(word2id=word2id,
                   id2word=id2word,
                   confusion_matrix=confusion_matrix,
                   meta_info=meta_info)

    @classmethod
    def from_saved(cls, part, side, noise_type):
        saved_dpath = cls.get_save_dpath(part=part, side=side, noise_type=noise_type)
        confusions_npz = np.load(os.path.join(saved_dpath, "confusions.npz"))
        return cls(word2id=json.load(open(os.path.join(saved_dpath, "word2id.json"))),
                   id2word=json.load(open(os.path.join(saved_dpath, "id2word.json"))),
                   confusion_matrix=confusions_npz["matrix"],
                   meta_info=json.load(open(os.path.join(saved_dpath, "meta_info.json"))))

    def save(self):
        save_dpath = self.get_save_dpath(part=self.part, side=self.side, noise_type=self.noise_type)
        os.makedirs(save_dpath, exist_ok=True)
        json.dump(self.word2id, open(os.path.join(save_dpath, "word2id.json"), "w"), indent=4)
        json.dump(self.id2word, open(os.path.join(save_dpath, "id2word.json"), "w"), indent=4)
        json.dump(self.meta_info, open(os.path.join(save_dpath, "meta_info.json"), "w"), indent=4)
        np.savez(os.path.join(save_dpath, 'confusions'), matrix=self.confusion_matrix)

    def _encode_token(self, word: str) -> int:
        return self.word2id[word]

    def _decode_token(self, id: int) -> str:
        return self.id2word[id]

    def _confuse_token(self, src_token: spacy.tokens.Token, correct_weight: float) ->Tuple[str, str]:
        # check src_token has only alphabet
        if self.alpha_only and not src_token.is_alpha:
            return src_token.text, OPETYPE.EQUAL

        src_token = src_token.text

        # except out of vocabulary
        try:
            src_id = self._encode_token(src_token)
        except KeyError:
            return src_token, OPETYPE.EQUAL

        # simulate error
        freqs = self.confusion_matrix[src_id]
        if not freqs.sum():
            freqs[src_id] = 1.
        weights = np.ones_like(freqs, dtype=np.float64)
        weights[src_id] = correct_weight
        freqs_weighted = freqs * weights
        probs = freqs_weighted / freqs_weighted.sum()
        new_id = np.random.choice(np.arange(probs.size), 1, p=probs)[0]
        new_token = self._decode_token(new_id)
        if src_token == new_token:
            confusion_type = OPETYPE.EQUAL
        elif new_token == self.del_word:
            confusion_type = OPETYPE.DELETION
        else:
            confusion_type = OPETYPE.SUBSTITUTION
        return new_token, confusion_type

    def apply_error(self, src_text, correct_weight=1.):
        if correct_weight < 1.:
            logger.warning(f"correct_weight is deprecated")
        doc, tokens = tokenize(src_text)
        confusions = []
        new_tokens = []
        for token in doc:
            new_token, opetype = self._confuse_token(token, correct_weight)
            confusions.append(opetype)
            if opetype != OPETYPE.DELETION:
                new_tokens.append(new_token)
                
        new_text = detokenize(new_tokens)
        return new_text, confusions