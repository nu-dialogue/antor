import os
import re
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

def get_optimizer_scheduler(model, learning_rate, adam_epsilon, weight_decay, warmup_steps, total_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler

def get_last_checkpoint_path(parent_checkpoint_dpath, use_final=False):
    files = os.listdir(parent_checkpoint_dpath)
    last_checkpoint_iter_id = -1
    last_checkpoint_path = ""
    for name in files:
        path = os.path.join(parent_checkpoint_dpath, name)
        if not os.path.isdir(path):
            continue
        iter_id = os.path.splitext(name)[1].strip(".")
        if iter_id == "final":
            if use_final:
                last_checkpoint_path = path
                break
        elif iter_id.isdecimal():
            iter_id = int(iter_id)
            if last_checkpoint_iter_id < iter_id:
                last_checkpoint_iter_id = iter_id
                last_checkpoint_path = path
        else:
            raise ValueError(f"Checkpoint's name '{name}' is incorrect format")
    if not last_checkpoint_path:
        raise RuntimeError(f"There is no available checkpoint in {parent_checkpoint_dpath}")
    return last_checkpoint_path

def make_act_sequence(act_bos_token, action):
    act_seq = " " + act_bos_token + ", ".join([f"{i}-{d}+{s}*{v}" for i,d,s,v in action])
    return act_seq

def make_resp_sequence(resp_bos_token, txt=None, eos_token=None):
    resp_seq = " " + resp_bos_token
    if txt is not None:
        assert eos_token is not None
        resp_seq += txt.strip() + eos_token
    return resp_seq

def split_act_sequence(seq, act_bos_token, resp_bos_token):
    act_seq, resp_seq = seq.replace(act_bos_token, "").split(resp_bos_token, 1)
    action = [re.split(r'[-\+\*]', a_seq) for a_seq in act_seq.split(", ")]
    return action, resp_seq