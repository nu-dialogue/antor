import json
import os
from distutils.util import strtobool
from collections import defaultdict
from tqdm import tqdm
import nltk
import wandb
import time
import numpy as np
import torch
import argparse

from common_utils.path import ROOT_DPATH
from common_utils.random_seed import set_seed
from common_utils.multiwoz_data import MultiWOZData
from sys_nlg.gpt2.model import build_gpt2
from sys_nlg.gpt2.nlg import GPT2NLG
from ppo_utils.ppo_updator import PPOUpdator, Rollouts
from ppo_utils.ppo_train_data import PPOTrainData, ActionIDF
from ppo_utils.reward import da_accuracy, ComputeReward
from experiments.speech_error_simulation.error_simulator import SpeechErrorSimulator

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_config(args):
    config = {
        "gpt2_config": {
            "pretrained_model_dpath": os.path.join(ROOT_DPATH, "sys_nlg/gpt2/outputs", args.checkpoints_dname, args.lm_name),
            "ref_model_dpath": os.path.join(ROOT_DPATH, "sys_nlg/gpt2/outputs",  args.checkpoints_dname, args.lm_name),
            "tokenizer_name": "gpt2",
            "lm_task_type": os.path.splitext(args.lm_name)[0],
            "act_bos_token": "[ACT]",
            "resp_bos_token": "[RSP]",
            "separate_vf": args.separate_vf
        },
        "ppo_config": {
            "batch_size": args.batch_size,
            "forward_batch_size": 128,
            "minibatch_size": 1,
            "ppo_epochs": args.ppo_epochs,
            "max_length": args.max_length,
            "lr": args.lr,
            "lr_linear_decay": args.lr_linear_decay,
            "gamma": args.gamma,
            "lam": args.lam,
            "cliprange": args.cliprange,
            "cliprange_value":args.cliprange_value,
            "vf_coef": args.vf_coef,
            "target_kl": args.target_kl,
            "init_kl_coef": args.init_kl_coef,
            "horizon": args.horizon,
            "temperature": args.temperature
        },
        "user_nlu_config": {
            "nlu_name": args.nlu_name,
            "nlu_model_name": args.nlu_model_name
        },
        "train_config": {
            "random_seed": args.random_seed,
            "total_iterations": args.total_iterations,
            "iterations_vf_pretrain": args.iterations_vf_pretrain,
            "reward_type": args.reward_type,
            "action_idf_weighted": args.action_idf_weighted,
            "save_checkpoint": args.save_checkpoint,
            "save_dpath": os.path.join(ROOT_DPATH, "experiments/ppo/outputs", args.run_id),
        },
        "noise_config": {
            "apply_noise": args.apply_noise,
            "part": args.noise_part,
            "side": args.noise_side,
            "noise_type": args.noise_type
        }
    }
    return config

def main(project_id, run_id, config):
    gpt2_config = config["gpt2_config"]
    ppo_config = config["ppo_config"]
    user_nlu_config = config["user_nlu_config"]
    train_config = config["train_config"]
    noise_config = config["noise_config"]
    
    os.makedirs(train_config["save_dpath"]) # do not overwrite
    json.dump(config, open(os.path.join(train_config["save_dpath"], "config.json"), "w"), indent=4)

    if train_config["random_seed"] is not None:
        set_seed(train_config["random_seed"])

    tokenizer, policy_gpt2, value_gpt2, ref_policy_gpt2  = build_gpt2(gpt2_config)

    s_nlg = GPT2NLG(gpt2=policy_gpt2, tokenizer=tokenizer,
                    lm_task_type=gpt2_config["lm_task_type"],
                    act_bos_token=gpt2_config["act_bos_token"],
                    resp_bos_token=gpt2_config["resp_bos_token"])
    s_ref_nlg = GPT2NLG(gpt2=ref_policy_gpt2, tokenizer=tokenizer,
                        lm_task_type=gpt2_config["lm_task_type"],
                        act_bos_token=gpt2_config["act_bos_token"],
                        resp_bos_token=gpt2_config["resp_bos_token"])

    ppo_updator = PPOUpdator(policy_model=policy_gpt2,
                             value_model=value_gpt2,
                             ref_policy_model=ref_policy_gpt2,
                             total_iterations=train_config["total_iterations"],
                             ppo_config=ppo_config)

    if noise_config["apply_noise"]:
        speech_error_simulator = SpeechErrorSimulator.from_saved(part=noise_config["part"],
                                                                 side=noise_config["side"],
                                                                 noise_type=noise_config["noise_type"])

    if user_nlu_config["nlu_name"] == "bert":
        from user_nlu.joint_bert.nlu import SimulatorBERTNLU
        u_nlu = SimulatorBERTNLU(config_fname=f"{user_nlu_config['nlu_model_name']}.json")
    elif user_nlu_config["nlu_name"] == "milu":
        from user_nlu.milu.nlu import UserMILU
        u_nlu = UserMILU(archive_dname=user_nlu_config["nlu_model_name"])
    elif user_nlu_config["nlu_name"] == "svm":
        from convlab2.nlu.svm.multiwoz import SVMNLU
        u_nlu = SVMNLU(mode="sys")

    def run_nlu(system_response):
        if noise_config["apply_noise"]:
            noised_system_response, _ = speech_error_simulator.apply_error(src_text=system_response)
            pred_action = u_nlu.predict(noised_system_response)
        else:
            noised_system_response = ""
            pred_action = u_nlu.predict(system_response)
        return noised_system_response, pred_action

    multiwoz_data = MultiWOZData()
    ppo_train_data = PPOTrainData(multiwoz_data=multiwoz_data,
                                  parts_used=["train"],
                                  batch_size=ppo_config["batch_size"],
                                  shuffle=True, infinite=True)
    action_idf = ActionIDF(multiwoz_data=multiwoz_data,
                           parts_used=["train"]) # , "val"]

    compute_reward = ComputeReward(reward_type=train_config["reward_type"],
                                   action_idf_weighted=train_config["action_idf_weighted"])

    run = wandb.init(project=project_id,
                     name=run_id,
                     config=config)
    best_score = {'reward': float('-inf')}
    for iteration_id in tqdm(range(train_config["total_iterations"])):
        table_log = defaultdict(list)
        ref_table_columns = ["ref/response", "ref/noised_response", "ref/f1", "ref/L_distance"]
        gen_table_columns = ["gen/response", "gen/noised_response", "gen/f1", "gen/L_distance"]
        test_table_columns = ["test/response", "test/noised_response", "test/f1", "test/L_distance"]

        env_log = defaultdict(list)
        timing = dict()
        t0 = time.time()

        rollouts = Rollouts(batch_size=ppo_config["batch_size"])

        # sample generation
        t = time.time()
        batch = ppo_train_data.sample_batch()
        ref_batch = []
        gen_batch = []
        test_batch = []
        for fbi in range(0, ppo_config["batch_size"], ppo_config["forward_batch_size"]):
            ref_batch += s_ref_nlg.batch_generate(batch=batch[fbi:fbi+ppo_config["forward_batch_size"]],
                                                  max_length=ppo_config["max_length"],
                                                  temperature=1.0,
                                                  do_sample=False)

            gen_batch_ = s_nlg.batch_generate(batch=batch[fbi:fbi+ppo_config["forward_batch_size"]],
                                              max_length=ppo_config["max_length"],
                                              temperature=ppo_config["temperature"],
                                              do_sample=True)
            gen_batch += gen_batch_
            for gen in gen_batch_:
                rollouts.insert_response(query_ids=gen['query_ids'].unsqueeze(0),
                                         response_ids=gen['response_ids'].unsqueeze(0),
                                         device=DEVICE)

            test_batch += s_nlg.batch_generate(batch=batch[fbi:fbi+ppo_config["forward_batch_size"]],
                                               max_length=ppo_config["max_length"],
                                               temperature=1.0,
                                               do_sample=False)
            timing['time/response_generation'] = time.time()-t
        timing['time/response_generation'] = time.time()-t

        # evaluate generation
        t = time.time()
        for bi in range(ppo_config["batch_size"]):
            action = batch[bi]["system_action"]
            # gt_response = batch[bi]["system_response"]
            ref_response = ref_batch[bi]["response_txt"].replace(tokenizer.eos_token, "")
            gen_response = gen_batch[bi]["response_txt"].replace(tokenizer.eos_token, "")
            test_response = test_batch[bi]["response_txt"].replace(tokenizer.eos_token, "")

            ref_noised_response, ref_pred_action = run_nlu(system_response=ref_response)
            gen_noised_response, gen_pred_action = run_nlu(system_response=gen_response)
            test_noised_response, test_pred_action = run_nlu(system_response=test_response)

            ref_acc = da_accuracy(true_action=action, pred_action=ref_pred_action)
            gen_acc = da_accuracy(true_action=action, pred_action=gen_pred_action)
            test_acc = da_accuracy(true_action=action, pred_action=test_pred_action)

            if ref_acc["tp_acts"]:
                ref_action_idfs = np.array([action_idf[gt_act] for gt_act in ref_acc["tp_acts"]])
            else:
                ref_action_idfs = np.array([0.])
            if gen_acc["tp_acts"]:
                gen_action_idfs =np.array([action_idf[gen_act] for gen_act in gen_acc["tp_acts"]])
            else:
                gen_action_idfs = np.array([0.])

            ref_tokens = ref_response.lower().split()
            gen_tokens = gen_response.lower().split()
            test_tokens = test_response.lower().split()

            ref_gen_nld = nltk.edit_distance(ref_tokens, gen_tokens) / max(len(ref_tokens), len(gen_tokens))
            ref_test_nld = nltk.edit_distance(ref_tokens, test_tokens) / max(len(ref_tokens), len(test_tokens))

            reward = compute_reward(ref_acc=ref_acc, ref_action_idfs=ref_action_idfs, ref_tokens=ref_tokens,
                                    gen_acc=gen_acc, gen_action_idfs=gen_action_idfs, gen_tokens=gen_tokens,
                                    ref_gen_nld=ref_gen_nld)
            rollouts.insert_reward(reward=torch.tensor([reward]), device=DEVICE)

            table_log["ref/response"].append(ref_response)
            table_log["gen/response"].append(gen_response)
            table_log["test/response"].append(test_response)
            table_log["ref/noised_response"].append(ref_noised_response)
            table_log["gen/noised_response"].append(gen_noised_response)
            table_log["test/noised_response"].append(test_noised_response)
            table_log["ref/f1"].append(ref_acc["f1"])
            table_log["gen/f1"].append(gen_acc["f1"])
            table_log["test/f1"].append(test_acc["f1"])
            table_log["ref/L_distance"].append(0.)
            table_log["gen/L_distance"].append(ref_gen_nld)
            table_log["test/L_distance"].append(ref_test_nld)

            env_log["reward"].append(reward)
            env_log["ref/f1"].append(ref_acc["f1"])
            env_log["gen/f1"].append(gen_acc["f1"])
            env_log["test/f1"].append(test_acc["f1"])
            env_log["ref/acc"].append(ref_acc["acc"])
            env_log["gen/acc"].append(gen_acc["acc"])
            env_log["test/acc"].append(test_acc["acc"])
            env_log["gen/length_increase"].append(len(ref_tokens)-len(gen_tokens))
            env_log["test/length_increase"].append(len(ref_tokens)-len(test_tokens))
        timing['time/response_evaluation'] = time.time()-t

        env_result = {'env/reward_mean': np.mean(env_log['reward']).item(),
                      'env/reward_std': np.std(env_log['reward']).item(),
                      'env/reward_dist': env_log['reward'],
                      'env/gen_f1': np.mean(env_log["gen/f1"]).item(),
                      'env/test_f1': np.mean(env_log["test/f1"]).item(),
                      'env/ref_f1': np.mean(env_log["ref/f1"]).item(),
                      'env/gen_acc': np.mean(env_log["gen/acc"]).item(),
                      'env/test_acc': np.mean(env_log["test/acc"]).item(),
                      'env/ref_acc': np.mean(env_log["ref/acc"]).item(),
                      'env/length_increase': np.mean(env_log["test/length_increase"]).item()}

        # save checkpoint
        if best_score["reward"] < env_result["env/reward_mean"]:
            best_score["reward"] = env_result["env/reward_mean"]
            if train_config["save_checkpoint"]:
                t = time.time()
                policy_gpt2.save_checkpoint(tokenizer=tokenizer,
                                            output_dpath=train_config["save_dpath"],
                                            prefix=f"ppo.{iteration_id}",
                                            eval_results=env_result)
                timing['time/checkpoint_save'] = time.time()-t
        
        # update by ppo
        t = time.time()
        if iteration_id < train_config["iterations_vf_pretrain"]:
            stats = ppo_updator.step(rollouts=rollouts, update_vf_only=True)
        else:
            stats = ppo_updator.step(rollouts=rollouts)
        timing['time/optimization'] = time.time()-t
        timing['time/epoch'] = time.time()-t0

        # logging
        ref_table_rows = [list(row) for row in zip(*(table_log[col] for col in ref_table_columns))]
        gen_table_rows = [list(row) for row in zip(*(table_log[col] for col in gen_table_columns))]
        test_table_rows = [list(row) for row in zip(*(table_log[col] for col in test_table_columns))]
        wandb.log({**env_result,
                   **stats,
                   **timing,
                   'table/ref': wandb.Table(columns=ref_table_columns, rows=ref_table_rows),
                   'table/gen': wandb.Table(columns=gen_table_columns, rows=gen_table_rows),
                   'table/test': wandb.Table(columns=test_table_columns, rows=test_table_rows)})

    # save final weight
    policy_gpt2.save_checkpoint(tokenizer=tokenizer,
                                output_dpath=train_config["save_dpath"],
                                prefix=f"ppo.final",
                                eval_results=env_result)

if __name__ == "__main__":
    def none_or(type_):
        def func(x):
            if x in ["None", "none"]:
                return None
            else:
                return type_(x)
        return func
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str)
    parser.add_argument("--run_id", type=str)

    parser.add_argument("--checkpoints_dname", type=str, default="checkpoints")
    parser.add_argument("--lm_name", type=str, default="act_resp.4")
    
    parser.add_argument("--separate_vf", type=strtobool, default=False)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--lr_linear_decay", type=strtobool, default=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--target_kl", type=none_or(float), default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--nlu_name", type=str, default="milu")
    parser.add_argument("--nlu_model_name", type=str, default="full-sys")

    parser.add_argument("--random_seed", type=none_or(int), default=None)
    parser.add_argument("--total_iterations", type=int, default=60)
    parser.add_argument("--iterations_vf_pretrain", type=int, default=0)
    parser.add_argument("--reward_type", type=str, default="F1")
    parser.add_argument("--action_idf_weighted", type=strtobool, default=True)
    parser.add_argument("--save_checkpoint", type=strtobool, default=True)

    parser.add_argument("--apply_noise", type=strtobool, default=False)
    parser.add_argument("--noise_type", type=str, default="background")
    parser.add_argument("--noise_part", type=str, default="train")
    parser.add_argument("--noise_side", type=str, default="sys")
    
    args = parser.parse_args()
    config = get_config(args)

    main(project_id=args.project_id, run_id=args.run_id, config=config)