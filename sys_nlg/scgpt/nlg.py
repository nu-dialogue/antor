from os.path import join, dirname, abspath, isdir
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from convlab2.nlg.scgpt.utils import tuple2seq

from common_utils.multiwoz_data import remove_ws_before_punctuation

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SCGPTNLG:
    def __init__(self, checkpoint_name="multiwoz_finetuned/checkpoint-20000") -> None:
        checkpoint_dpath = join(dirname(abspath(__file__)), checkpoint_name)
        if not isdir(checkpoint_dpath):
            raise FileNotFoundError(f"Pretrained checkpoint of scgpt should be in {checkpoint_dpath}")
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint_dpath)
        self.model.to(DEVICE)
        self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dpath)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, action, ret_w_eos_token=False, **kwargs):
        raw_text = tuple2seq(action)
        raw_text += " &"
        input_ids = self.tokenizer.encode(raw_text, return_tensors="pt",
                                          add_special_tokens=False)
        batch_size, input_len = input_ids.size()
        outputs = self.model.generate(
            input_ids=input_ids.to(DEVICE),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=kwargs["max_length"],
            top_k=0, top_p=1.0, num_beams=1, 
            do_sample=kwargs["do_sample"],
            temperature=kwargs["temperature"])
        gen_ids = outputs[0, input_len:]
        text = self.tokenizer.decode(gen_ids, clean_up_tokenization_spaces=True)
        text = text.split('& ')[-1]
        if not ret_w_eos_token:
            text = text[: text.find(self.tokenizer.eos_token) if self.tokenizer.eos_token else None]
        text = remove_ws_before_punctuation(text)
        return text