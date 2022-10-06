import json
import os
import argparse
from logging import getLogger

from sys_nlg.gpt2.utils import get_last_checkpoint_path
from sys_nlg.gpt2.model import build_gpt2
from sys_nlg.gpt2.nlg import GPT2NLG
from experiments.evaluate_model.core import evaluate_nlg
from common_utils.path import ROOT_DPATH
from common_utils.multiwoz_data import MultiWOZData
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

def config_from_run_id(run_id, multiwoz_data_part="test"):
    # 0. load ppo train config
    trained_dpath = os.path.join(ROOT_DPATH, "experiments/ppo/outputs", run_id)
    train_config_fpath = os.path.join(trained_dpath, "config.json")
    config = json.load(open(train_config_fpath))
    
    # 1. update gpt2 checkpoint path
    checkpoint_path = get_last_checkpoint_path(trained_dpath, use_final=True)
    logger.info(f"Checkpoint {checkpoint_path} will be used")
    config["gpt2_config"]["pretrained_model_dpath"] = checkpoint_path
    config["gpt2_config"]["ref_model_dpath"] = checkpoint_path

    # 2. add evaluation config
    config["sys_nlg_config"] = {
        "nlg_name": "gpt2ppo",
        "max_length": config["ppo_config"]["max_length"]
    }
    config["evaluate_config"] = {
        "random_seed": config["train_config"]["random_seed"],
        "multiwoz_data_part": multiwoz_data_part,
        "save_dpath": os.path.join(ROOT_DPATH, "experiments/evaluate_model/outputs/ppo", run_id)
    }
    return config


if __name__ == "__main__":
    # list run_id
    # ls -ls ../ppo/outputs | awk 'NR>1{printf("run_id=\""$10"\"\n")}'
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--multiwoz_data_part", choices=MultiWOZData.parts, default="test")
    args = parser.parse_args()

    config = config_from_run_id(run_id=args.run_id,
                                multiwoz_data_part=args.multiwoz_data_part)

    # system nlg
    gpt2_config = config["gpt2_config"]
    tokenizer, policy_gpt2, _, _  = build_gpt2(gpt2_config)
    s_nlg = GPT2NLG(gpt2=policy_gpt2, tokenizer=tokenizer,
                    lm_task_type=gpt2_config["lm_task_type"],
                    act_bos_token=gpt2_config["act_bos_token"],
                    resp_bos_token=gpt2_config["resp_bos_token"])
    
    evaluate_nlg(s_nlg=s_nlg, config=config)