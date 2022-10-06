import os
import torch
import argparse

from common_utils.path import ROOT_DPATH
from common_utils.multiwoz_data import MultiWOZData
from sys_nlg.gpt2.model import GPT2HeadWithValueHeadModel
from transformers import GPT2Tokenizer
from sys_nlg.gpt2.sl_trainer import SupervisedTrainer

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def config_from_args(args):
    config = {
        "sl_config": {
            'train_size_ratio': 1.0,
            'checkpoints_output_dpath': os.path.join(ROOT_DPATH, 'sys_nlg/gpt2/outputs/checkpoints'),

            'batch_size': args.batch_size,
            'eval_batch_size': args.eval_batch_size,
            'epoch_num': args.epoch_num,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'max_grad_norm': args.max_grad_norm,
            'weight_decay': args.weight_decay,
            'learning_rate': args.learning_rate,
            'adam_epsilon': args.adam_epsilon,
            'warmup_steps': args.warmup_steps,
            'report_interval': args.report_interval
        },
        "gpt2_config": {
            "pretrained_dpath": "gpt2",
            "tokenizer_name": "gpt2",
            "lm_task_type": "act_resp",
            "act_bos_token": "[ACT]",
            "resp_bos_token": "[RSP]",
        }
    }
    return config

def main(config):
    sl_config = config["sl_config"]
    gpt2_config = config["gpt2_config"]

    model = GPT2HeadWithValueHeadModel.from_pretrained(gpt2_config["pretrained_dpath"])
    model.to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_config["tokenizer_name"])
    multiwoz_data = MultiWOZData(rm_ws_before_punc=True)

    sl_trainer = SupervisedTrainer(model, tokenizer,
                                   multiwoz_data=multiwoz_data, sl_config=sl_config,
                                   lm_task_type=gpt2_config["lm_task_type"],
                                   act_bos_token=gpt2_config["act_bos_token"],
                                   resp_bos_token=gpt2_config["resp_bos_token"])
    sl_trainer.load_dataset()

    sl_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epoch_num", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--report_interval", type=int, default=500)
    args = parser.parse_args()

    config = config_from_args(args)
    main(config)