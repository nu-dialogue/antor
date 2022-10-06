import os
import json
import math
from copy import deepcopy
from typing import NamedTuple, List
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizer

from common_utils.multiwoz_data import MultiWOZData
from sys_nlg.gpt2.utils import make_act_sequence, make_resp_sequence

from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

def prepare_act_resp_datasets(multiwoz_data: MultiWOZData, tokenizer: PreTrainedTokenizer,
                              act_bos_token, resp_bos_token, train_size_ratio):
    def _build_examples(data):
        examples = []
        for i, d in enumerate(data):
            act_seq = make_act_sequence(act_bos_token=act_bos_token,
                                        action=d["system_action"])
            resp_seq = make_resp_sequence(resp_bos_token=resp_bos_token,
                                          txt=d["system_response"],
                                          eos_token=tokenizer.eos_token)
            examples.append({
                "indices": i,
                "act_seq": act_seq,
                "resp_seq": resp_seq
            })
        return examples

    act_resp_data = multiwoz_data.get_act_resp_data(parts=["train", "val"])

    train_size = math.ceil(len(act_resp_data["train"]) * train_size_ratio)
    train_data = act_resp_data["train"][:train_size]
    valid_data = act_resp_data["val"]

    train_examples = _build_examples(data=train_data)
    valid_examples = _build_examples(data=valid_data)
    datasets = {"train": ActionResponseDataset(examples=train_examples, tokenizer=tokenizer),
                "val": ActionResponseDataset(examples=valid_examples, tokenizer=tokenizer)}
    return datasets

class ActionResponseDataset(Dataset):
    def __init__(self, examples, tokenizer: PreTrainedTokenizer) -> None:
        self.indices = []

        self.act_ids = []
        self.resp_ids = []
        
        self.act_txt = []
        self.resp_txt = []

        for d in examples:
            self.indices.append(d["indices"])

            act_ids = tokenizer.encode(d["act_seq"])
            resp_ids = tokenizer.encode(d["resp_seq"])
            if act_ids+resp_ids != tokenizer.encode(d["act_seq"]+d["resp_seq"]):
                open("tmp.out", "a").write(f'{d["act_seq"]} + {d["resp_seq"]}\n')

            self.act_ids.append(act_ids)
            self.resp_ids.append(resp_ids)

            self.act_txt.append(d["act_seq"]) # for debug
            self.resp_txt.append(d["resp_seq"]) # for debug

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, index: int):
        return {"indices": self.indices[index],
                "act_ids": self.act_ids[index],
                "resp_ids": self.resp_ids[index],
                "act_txt": self.act_txt[index],
                "resp_txt": self.resp_txt[index]}

class Batch(NamedTuple):
    indices: torch.LongTensor
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    resp_masks: torch.LongTensor
    input_txt: List[str]

def act_resp_collate(batch, pad_token_id, ignore_index):
    indices = []
    act_resp_ids = []
    act_resp_txt = []
    resp_masks = []
    for b in batch:
        indices.append(torch.tensor(b["indices"], dtype=torch.long))
        act_resp_ids.append(torch.tensor(b["act_ids"]+b["resp_ids"], dtype=torch.long))
        resp_masks.append(torch.tensor([0]*len(b["act_ids"])+[1]*len(b["resp_ids"]), dtype=torch.long))
        act_resp_txt.append(b["act_txt"]+b["resp_txt"])
    return Batch(indices=torch.stack(indices),
                 input_ids=pad_sequence(act_resp_ids, batch_first=True, padding_value=pad_token_id),
                 labels = pad_sequence(act_resp_ids, batch_first=True, padding_value=ignore_index),
                 resp_masks=pad_sequence(resp_masks, batch_first=True, padding_value=0),
                 input_txt=act_resp_txt)

def get_dataloader(lm_task_type, dataset, batch_size, pad_token_id, ignore_index, is_train):
    if pad_token_id == ignore_index:
        logger.warning.warn(f"Padding idx and ignore idx are the same value ({pad_token_id}).")
    def collate_fn(batch):
        if lm_task_type == "act_resp":
            return act_resp_collate(batch=batch, pad_token_id=pad_token_id, ignore_index=ignore_index)
        else:
            raise ValueError

    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader