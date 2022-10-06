
class ComputeReward:
    def __init__(self, reward_type, action_idf_weighted=False) -> None:
        self.reward_type = reward_type
        self.action_idf_weighted_reward = action_idf_weighted

    def __call__(self, **kwds) -> float:
        ref_acc = kwds["ref_acc"]
        gen_acc = kwds["gen_acc"]
        ref_action_idfs = kwds["ref_action_idfs"]
        gen_action_idfs = kwds["gen_action_idfs"]
        ref_tokens = kwds["ref_tokens"]
        gen_tokens = kwds["gen_tokens"]
        ref_gen_nld = kwds["ref_gen_nld"]

        ref_tp = len(ref_acc["tp_acts"])
        ref_fp = len(ref_acc["fp_acts"])
        ref_fn = len(ref_acc["fn_acts"])
        ref_f1 = ref_acc["f1"]
        gen_tp = len(gen_acc["tp_acts"])
        gen_fp = len(gen_acc["fp_acts"])
        gen_fn = len(gen_acc["fn_acts"])
        gen_f1 = gen_acc["f1"]

        if self.action_idf_weighted_reward:
            ref_tp = ref_action_idfs.sum()
            ref_f1 *= ref_action_idfs.mean()
            gen_tp = gen_action_idfs.sum()
            gen_f1 *= gen_action_idfs.mean()

        f1_increase = gen_f1 - ref_f1
        ref_len = len(ref_tokens)
        gen_len = len(gen_tokens)
        length_increase = gen_len-ref_len
        
        if self.reward_type == "raw_score":
            reward = (gen_tp-(gen_fp+gen_fn))/10

        elif self.reward_type == "raw_score_increase":
            reward = (gen_tp-(gen_fp+gen_fn)) - (ref_tp-(ref_fp+ref_fn))

        elif self.reward_type == "F1":
            reward = gen_f1
        
        elif self.reward_type == "F1_increase":
            if gen_f1 == 0:
                reward = -1
            else:
                reward = 2*f1_increase
        
        elif self.reward_type == "F1XL_distance":
            if gen_f1 == 0:
                reward = -1
            elif ref_gen_nld == 0:
                reward = -0.5
            else:
                reward = gen_f1 * min(10*ref_gen_nld, 1.0)
        
        elif self.reward_type == "F1_increaseXL_distance":
            if f1_increase < 0:
                reward = f1_increase * min(10*ref_gen_nld, 1.0)
            else:
                reward = min(10*ref_gen_nld, 1.0)
        
        elif self.reward_type == "F1Xlength_decrease":
            if gen_f1 == 0:
                reward = -1
            else:
                reward = gen_f1 * (-length_increase/ref_len)
        
        elif self.reward_type == "F1_increaseXlength_decrease":
            if gen_f1 == 0:
                reward = -1
            elif f1_increase < 0:
                reward = 2*f1_increase
            else:
                reward = -length_increase/ref_len
        
        elif self.reward_type == "F1Xlength_rate":
            if gen_f1 == 0:
                reward = -1
            else:
                reward = gen_f1 * ref_len/gen_len
        
        elif self.reward_type == "F1_increaseXlength_rate":
            if gen_f1 == 0 or gen_len == 0:
                reward = -1
            elif f1_increase < 0:
                reward = 2*f1_increase
            else:
                reward = ref_len/gen_len

        elif self.reward_type == "question":
            if gen_tokens and gen_tokens[-1] == "?":
                reward = 2 * gen_f1
            else:
                reward = -1
        else:
            raise NotImplementedError
        return reward

def da_accuracy(true_action, pred_action):
    tp_acts, fn_acts, fp_acts = [], [], []
    lowered_true_actions = []
    lowered_pred_actions = []
    for true_act in true_action:
        lowered_true_actions.append([true_act[0], true_act[1], true_act[2], true_act[3].lower()])
    for pred_act in pred_action:
        lowered_pred_actions.append([pred_act[0], pred_act[1], pred_act[2], pred_act[3].lower()])
    
    for true_act in true_action:
        if [true_act[0], true_act[1], true_act[2], true_act[3].lower()] in lowered_pred_actions:
            tp_acts.append(true_act)
        else:
            fn_acts.append(true_act)
    for pred_act in pred_action:
        if [pred_act[0], pred_act[1], pred_act[2], pred_act[3].lower()] not in lowered_true_actions:
            fp_acts.append(pred_act)

    tp, fn, fp = len(tp_acts), len(fn_acts), len(fp_acts)
    acc, recall, precision, f1 = 0, 0, 0, 0
    if tp + fn:
        recall = tp / (tp + fn)
    if tp + fp:
        precision = tp / (tp + fp)
    if recall + precision:
        f1 = (2*recall*precision) / (recall+precision)
    if tp + fn + fp:
        acc = tp / (tp + fn + fp)

    result = {"tp_acts":tp_acts, "fn_acts":fn_acts, "fp_acts":fp_acts,
              "recall":recall, "precision":precision, "f1":f1,
              "acc": acc}
    return result
