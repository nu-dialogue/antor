import torch
from sys_nlg.nlg import BaseNLG
from sys_nlg.gpt2.utils import make_act_sequence, make_resp_sequence

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GPT2NLG(BaseNLG):
    def __init__(self, gpt2, tokenizer, lm_task_type, act_bos_token, resp_bos_token) -> None:
        super().__init__()
        self.gpt2 = gpt2
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm_task_type = lm_task_type
        self.act_bos_token = act_bos_token
        self.resp_bos_token = resp_bos_token

    def _make_input_seq(self, system_action):
        act_seq = make_act_sequence(act_bos_token=self.act_bos_token,
                                    action=system_action)
        resp_seq = make_resp_sequence(resp_bos_token=self.resp_bos_token)
        if self.lm_task_type == "act_resp":
            input_seq = act_seq + resp_seq
        else:
            raise ValueError
        return input_seq

    def generate(self, action, ret_w_eos_token=False, **kwargs) -> str:
        input_seq = self._make_input_seq(system_action=action)
        input_ids = self.tokenizer.encode(input_seq, return_tensors="pt")
        batch_size, query_len = input_ids.size()
        outputs = self.gpt2.generate(
            input_ids=input_ids.to(DEVICE),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=kwargs["max_length"],
            top_k=0, top_p=1.0, num_beams=1, 
            do_sample=kwargs["do_sample"], temperature=kwargs["temperature"])
        gen_ids = outputs[0, query_len:]
        gen_txt = self.tokenizer.decode(gen_ids)
        if not ret_w_eos_token:
            gen_text = gen_txt.replace(self.tokenizer.eos_token, "")
        return gen_text

    def batch_generate(self, batch, ret_w_eos_token=True, **kwargs):
        """
        batch = [
            {"system_action": List[List[str, str, str, str], ...], "system_response": str},
            ...
        ]
        """
        input_txt_list = []
        for b in batch:
            input_seq = self._make_input_seq(system_action=b["system_action"])
            input_txt_list.append(input_seq)
        
        # batch generation
        # https://github.com/huggingface/transformers/pull/7552#issue-497255933
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(input_txt_list, padding='longest',
                                      return_tensors="pt", return_length=True)
        self.tokenizer.padding_side = "right"
        act_gen_ids = self.gpt2.generate(
            input_ids=model_inputs["input_ids"].to(DEVICE),
            attention_mask=model_inputs["attention_mask"].to(DEVICE),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=kwargs["max_length"],
            # min_length=0,
            top_k=0, top_p=1.0, num_beams=1, 
            do_sample=kwargs["do_sample"], temperature=kwargs["temperature"])

        batch_size, query_max_len = model_inputs["input_ids"].size()
        gen_ids = act_gen_ids[:, query_max_len:]
        eos_offset = int(self.tokenizer.eos_token_id == self.tokenizer.pad_token_id)
        gen_batch = []
        for i in range(batch_size):
            query_ids = model_inputs["input_ids"][i, -model_inputs["length"][i]:]

            gen_pad_len = gen_ids[i].eq(self.tokenizer.pad_token_id).sum()
            if gen_pad_len <= eos_offset: # eos_tokenとpad_tokenが同じときは1つ以下
                pad_start_idx = None
            else:
                pad_start_idx = -gen_pad_len + eos_offset
            response_ids = gen_ids[i, :pad_start_idx]
            response_txt = self.tokenizer.decode(response_ids)
            if not ret_w_eos_token:
                response_txt = response_txt.replace(self.tokenizer.eos_token, "")
            
            gen_batch.append({
                "query_ids": query_ids,
                "response_ids": response_ids,
                "response_txt": response_txt
            })
        return gen_batch