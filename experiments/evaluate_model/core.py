import json
import os
from logging import getLogger
import pandas as pd
from tqdm import tqdm
import jiwer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from common_utils.random_seed import set_seed
from common_utils.multiwoz_data import MultiWOZData
from experiments.speech_error_simulation.error_simulator import SpeechErrorSimulator
from ppo_utils.reward import da_accuracy
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

def evaluate_nlg(s_nlg, config):
    sys_nlg_config = config["sys_nlg_config"]
    user_nlu_config = config["user_nlu_config"]
    noise_config = config["noise_config"]
    evaluate_config = config["evaluate_config"]

    if evaluate_config["random_seed"] is not None:
        set_seed(evaluate_config["random_seed"])

    # user nlu
    if user_nlu_config["nlu_name"] == "bert":
        from user_nlu.joint_bert.nlu import SimulatorBERTNLU
        u_nlu = SimulatorBERTNLU(config_fname=f"{user_nlu_config['nlu_model_name']}.json")
    elif user_nlu_config["nlu_name"] == "milu":
        from user_nlu.milu.nlu import UserMILU
        u_nlu = UserMILU(archive_dname=user_nlu_config["nlu_model_name"])
    else:
        raise NotImplementedError

    if noise_config["apply_noise"]:
        speech_error_simulator = SpeechErrorSimulator.from_saved(part=noise_config["part"],
                                                                 side=noise_config["side"],
                                                                 noise_type=noise_config["noise_type"])

    # evaluate model on multiwoz data
    ## accuracy and wer
    part = evaluate_config["multiwoz_data_part"]
    multiwoz_data = MultiWOZData(True)
    act_resp_data = multiwoz_data.get_act_resp_data(parts=[part])
    result = []
    for turn in tqdm(act_resp_data[part]):
        ref_text = turn["system_response"]
        action = turn["system_action"]
        gen_text = s_nlg.generate(action=action,
                                  max_length=sys_nlg_config["max_length"],
                                  do_sample=False, temperature=1.0)
                                  
        if noise_config["apply_noise"]:
            noised_text, _ = speech_error_simulator.apply_error(src_text=gen_text)
            try:
                wer = jiwer.wer(truth=gen_text, hypothesis=noised_text)
            except ValueError:
                wer = None
            pred_action = u_nlu.predict(noised_text)
        else:
            noised_text = ""
            wer = None
            pred_action = u_nlu.predict(gen_text)
            
        acc = da_accuracy(true_action=action, pred_action=pred_action)
        result.append({
            "ref_text": ref_text,
            "action": action,
            "gen_text": gen_text,
            "noised_text": noised_text,
            "noised_wer": wer,
            **acc
        })
    ## BLEU score
    refs = [[r["ref_text"].lower().split()] for r in result]
    hyps = [r["gen_text"].lower().split() for r in result]
    bleu_score = corpus_bleu(list_of_references=refs,
                             hypotheses=hyps,
                             smoothing_function=SmoothingFunction().method1)
    
    # summrize evaluation results
    result_summary = pd.DataFrame(result).mean(numeric_only=True).to_dict()
    result_summary["bleu_score"] = bleu_score

    # save evaluation results
    save_dpath = evaluate_config["save_dpath"]
    os.makedirs(save_dpath) # do not use exist_ok=True

    save_config_fpath = os.path.join(save_dpath, "config.json")
    json.dump(config, open(save_config_fpath, "w"), indent=4)
    
    save_result_fpath = os.path.join(save_dpath, "result.json")
    json.dump(result, open(save_result_fpath, "w"), indent=4)
    
    save_result_summary_fpath = os.path.join(save_dpath, "result_summary.json")
    json.dump(result_summary, open(save_result_summary_fpath, "w"), indent=4)