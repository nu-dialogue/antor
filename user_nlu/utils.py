import os
import json
import random

from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

def count_turns_to_use(dialogs_to_use):
    num_turns_to_use = 0
    num_total_turns = 0
    for turns in dialogs_to_use.values():
        num_turns_to_use += sum(turns)
        num_total_turns += len(turns)
    return num_turns_to_use, num_total_turns

def get_dialogs_to_use(part, data, vocab_level_tolerance, vocab_level_by_turn_dpath):
    dialogs_to_use = {dial_name: [True]*len(data[dial_name]["log"]) for dial_name in data}
    
    # 1. resize data based on data_size_ratio
    # if data_size_ratio < 1.0 and part in ["train", "val"]:
    #     data_size = int(len(data) * data_size_ratio)
    #     names_dial_to_use = random.sample(list(data), k=data_size)
    #     for dial_name in dialogs_to_use:
    #         if dial_name not in names_dial_to_use:
    #             dialogs_to_use[dial_name] = [False]*len(dialogs_to_use[dial_name])
    #     num_turns_to_use, num_total_turns = count_turns_to_use(dialogs_to_use=dialogs_to_use)
    #     logger.info(f"Resized {part} data: {num_turns_to_use}/{num_total_turns}")
    # else:
    #     logger.info(f"Didn't resize {part} data")

    # 2. Remove turn based on vocabulary level
    num_user, num_sys = 0, 0
    if vocab_level_tolerance and part in ["train", "val"]:
        vocab_level_by_turn_fpath = os.path.join(vocab_level_by_turn_dpath, f"{part}.json")
        vocab_level_by_turn = json.load(open(vocab_level_by_turn_fpath))
        for dial_name, turn_vocab_level in vocab_level_by_turn.items():
            for i, vlevel in enumerate(turn_vocab_level):
                if vlevel not in vocab_level_tolerance:
                    dialogs_to_use[dial_name][i] = False
                else:
                    if i % 2 == 0:
                        num_user += 1
                    else:
                        num_sys += 1
        num_turns_to_use, num_total_turns = count_turns_to_use(dialogs_to_use=dialogs_to_use)
        logger.info(f"Filtered {part} data at vocab_level {vocab_level_tolerance}: " \
            f"{num_turns_to_use}(user={num_user},sys={num_sys})/{num_total_turns} will be used ")
    else:
        logger.info(f"Didn't filter {part} data at vocab_level")

    return dialogs_to_use