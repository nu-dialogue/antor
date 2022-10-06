from convlab2.nlg.sclstm.multiwoz import SCLSTM
from common_utils.multiwoz_data import remove_ws_before_punctuation

class SCLSTMNLG(SCLSTM):
    def __init__(self) -> None:
        super().__init__()
        self.unk_token = "UNK_token"
    
    def generate(self, action, **kwargs):
        self.args["beam_size"] = 1
        gen_txt = super().generate(action)
        gen_txt = gen_txt.replace(self.unk_token, "")
        gen_txt = remove_ws_before_punctuation(gen_txt)
        return gen_txt