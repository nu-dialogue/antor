from convlab2.nlu.jointBERT.multiwoz import BERTNLU

class SimulatorBERTNLU(BERTNLU):
    def __init__(self, config_fname):
        super().__init__(mode='sys', config_file=config_fname,
                         model_file=None)
                         