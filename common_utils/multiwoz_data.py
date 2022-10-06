import os
import re
from copy import deepcopy
from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)
from common_utils.file_reader import read_zipped_json
from common_utils.path import ROOT_DPATH

def remove_ws_before_punctuation(text, span_info=None):
    # if span_info is None:
    #     # https://stackoverflow.com/questions/20047387/remove-space-before-punctuation-javascript-jquery 
    #     return re.sub(r'\s+(\W)', r'\1', text)

    def update_span_info(span_info, i):
        span_info_ = []
        for *da, start, end in span_info:
            if i <= start:
                start -= 1
                end -= 1
            elif i <= end:
                end -= 1
            span_info_.append([*da, start, end])
        return span_info_

    tokens = text.split()
    new_tokens = []
    new_span_info = deepcopy(span_info)
    for i, token in enumerate(tokens):
        m = re.match(r'\W|n\'', token) # " ."  " ,"  " ?"  " n't"  " 're"
        if m is None or not new_tokens:
            new_tokens.append(token)
        else:
            new_tokens[-1] += token
            if new_span_info is not None:
                new_span_info = update_span_info(new_span_info, i)
                
    if new_span_info is not None:
        return " ".join(new_tokens), new_span_info
    else:
        return " ".join(new_tokens)
    

def make_tags(length, span_info):
    tags = []
    for i in range(length):
        for span in span_info:
            if i == span[3]:
                tags.append("B-" + span[0] + "+" + span[1])
                break
            if span[3] < i <= span[4]:
                tags.append("I-" + span[0] + "+" + span[1])
                break
        else:
            tags.append("O")
    return tags

def flatten_da(dialog_act):
    triples = []
    for intent, svs in dialog_act.items():
        domain, intent = intent.split("-")
        for slot, value in svs:
            triples.append([intent, domain, slot, value])
    return triples

class MultiWOZData:
    parts = ["train", "val", "test"]
    sides = ["sys", "user"]

    def __init__(self, rm_ws_before_punc=False):
        self.raw_data = {}
        for part in MultiWOZData.parts:
            multiwoz_data_fpath = os.path.join(ROOT_DPATH, f"ConvLab-2/data/multiwoz/{part}.json.zip")
            self.raw_data[part] = read_zipped_json(multiwoz_data_fpath, f'{part}.json')
        
        if rm_ws_before_punc:
            self._rm_ws_before_punc()

    def __getitem__(self, part):
        assert part in MultiWOZData.parts
        return self.raw_data[part]

    def iter_dialog_log(self, part, dial_name):
        for i, turn in enumerate(self.raw_data[part][dial_name]["log"]):
            is_sys = i%2
            side = "sys" if is_sys else "user"
            yield i, side, turn

    def _rm_ws_before_punc(self):
        logger.warn("After removing white spaces before punctuation, "
                    "span_info may become inconsistent.")
        for part in MultiWOZData.parts:
            for dial_name in self.raw_data[part]:
                for i, side, turn in self.iter_dialog_log(part=part, dial_name=dial_name):
                    text, span_info = remove_ws_before_punctuation(turn["text"], turn["span_info"])
                    turn["text"] = text
                    turn["span_info"] = span_info
                    
    # def split_dialogs(self, parts, num_dials_per_chunk):
    #     def chunks(l):
    #         for i in range(0, len(l), num_dials_per_chunk):
    #             yield l[i:i+num_dials_per_chunk]
    #     assert not set(parts) - set(self.parts)
    #     splited_dialogs = {}
    #     for part in parts:
    #         all_dialogs = list(self.raw_data[part])
    #         splited_dialogs[part] = list(chunks(all_dialogs))
    #         """
    #         splited_dialogs[part] = [
    #             ["dial_name 1", "dial_name 2", ..., "dial_name n"],
    #             ["dial_name n+1", "dial_name n+2", ..., "dial_name 2n"],
    #             ...
    #         ]
    #         """
    #     return splited_dialogs

    def get_act_resp_data(self, parts):
        assert not set(parts) - set(MultiWOZData.parts)

        act_resp_data = {}
        for part in parts:
            act_resp_data[part] = []
            for dial_name in self.raw_data[part]:
                for i, side, turn in self.iter_dialog_log(part=part, dial_name=dial_name):
                    if side == "sys":
                        act_resp_data[part].append({
                            "system_response": turn["text"],
                            "system_action": flatten_da(turn["dialog_act"])
                        })
        return act_resp_data

    def stats(self):
        stats = {}
        for part in MultiWOZData.parts:
            stats[part] = {"num_dials": 0,
                           "num_chars": 0,
                           "num_texts": 0,
                           "num_sys_chars": 0,
                           "num_sys_texts": 0,
                           "num_user_chars": 0,
                           "num_user_texts": 0}
            for dial_name in self.raw_data[part]:
                stats[part]["num_dials"] += 1
                for i, side, turn in self.iter_dialog_log(part=part, dial_name=dial_name):
                    stats[part]["num_chars"] += len(turn["text"])
                    stats[part]["num_texts"] += 1
                    stats[part][f"num_{side}_chars"] += len(turn["text"])
                    stats[part][f"num_{side}_texts"] += 1
        return stats
