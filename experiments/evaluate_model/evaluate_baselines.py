import os
import argparse
from logging import getLogger
from distutils.util import strtobool

from experiments.evaluate_model.core import evaluate_nlg
from common_utils.path import ROOT_DPATH
from common_utils.multiwoz_data import MultiWOZData
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

def get_config(args):
    if args.nlg_name == "gpt2":
        assert args.gpt2_checkpoint_dname is not None
        gpt2_checkpoint_dpath = os.path.join(ROOT_DPATH, "sys_nlg/gpt2/outputs/checkpoints", args.gpt2_checkpoint_dname)
        gpt2_config = {
            "pretrained_model_dpath": gpt2_checkpoint_dpath,
            "ref_model_dpath": gpt2_checkpoint_dpath,
            "tokenizer_name": "gpt2",
            "lm_task_type": "act_resp",
            "act_bos_token": "[ACT]",
            "resp_bos_token": "[RSP]",
            "separate_vf": False
        }
    else:
        gpt2_config = None

    config = {
        "gpt2_config": gpt2_config,
        "sys_nlg_config": {
            "nlg_name": args.nlg_name,
            "max_length": args.nlg_max_length
        },
        "user_nlu_config": {
            "nlu_name": args.nlu_name,
            "nlu_model_name": args.nlu_model_name
        },
        "noise_config": {
            "apply_noise": args.apply_noise,
            "part": args.noise_part,
            "side": args.noise_side,
            "noise_type": args.noise_type
        },
        "evaluate_config": {
            "random_seed": args.random_seed,
            "multiwoz_data_part": args.multiwoz_data_part,
            "save_dpath": os.path.join(ROOT_DPATH, "experiments/evaluate_model/outputs/baselines", args.evaluate_id)
        }
    }
    return config

if __name__ == "__main__":
    def none_or(type_):
        def func(x):
            if x in [None, "None", "none"]:
                return None
            else:
                return type_(x)
        return func

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate_id", type=str, required=True)
    parser.add_argument("--nlg_name", choices=["gpt2", "sclstm", "scgpt"], default="gpt2")
    parser.add_argument("--gpt2_checkpoint_dname", type=none_or(str), default=None)
    parser.add_argument("--nlg_max_length", type=int, default=256)
    parser.add_argument("--nlu_name", choices=["milu", "bert"], default="milu")
    parser.add_argument("--nlu_model_name", type=str, default="full-sys")
    parser.add_argument("--apply_noise", type=strtobool, default=True)
    parser.add_argument("--noise_type", type=str, default="background(0)")
    parser.add_argument("--noise_part", choices=MultiWOZData.parts, default="train")
    parser.add_argument("--noise_side", choices=MultiWOZData.sides, default="sys")
    parser.add_argument("--random_seed", type=none_or(int), default=None)
    parser.add_argument("--multiwoz_data_part", choices=MultiWOZData.parts, default="test")
    args = parser.parse_args()

    config = get_config(args)
    
    gpt2_config = config["gpt2_config"]
    sys_nlg_config = config["sys_nlg_config"]

    # system nlg
    if sys_nlg_config["nlg_name"] == "gpt2":
        from sys_nlg.gpt2.model import build_gpt2
        from sys_nlg.gpt2.nlg import GPT2NLG
        assert gpt2_config is not None
        tokenizer, policy_gpt2, _, _  = build_gpt2(gpt2_config)
        s_nlg = GPT2NLG(gpt2=policy_gpt2, tokenizer=tokenizer,
                        lm_task_type=gpt2_config["lm_task_type"],
                        act_bos_token=gpt2_config["act_bos_token"],
                        resp_bos_token=gpt2_config["resp_bos_token"])
    elif sys_nlg_config["nlg_name"] == "sclstm":
        from sys_nlg.sclstm.nlg import SCLSTMNLG
        s_nlg = SCLSTMNLG()
    elif sys_nlg_config["nlg_name"] == "scgpt":
        from sys_nlg.scgpt.nlg import SCGPTNLG
        s_nlg = SCGPTNLG()
    else:
        raise NotImplementedError

    evaluate_nlg(s_nlg=s_nlg, config=config)