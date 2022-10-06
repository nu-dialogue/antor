import os
import torch
from convlab2.nlu.milu.multiwoz import MILU
from user_nlu.milu import dataset_reader

from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

CURRENT_DPATH = os.path.dirname(__file__)
class UserMILU(MILU):
    def __init__(self, archive_dname):
        if "context" in archive_dname:
            context_size = 3
        else:
            context_size = 0
        archive_file = os.path.join(CURRENT_DPATH, "outputs", archive_dname, "model.tar.gz")
        if not os.path.isfile(archive_file):
            raise FileExistsError(f"MILU archive file {archive_file} does not exist")
        super().__init__(archive_file=archive_file, context_size=context_size, model_file=None)
        
    def predict(self, utterance, context=...):
        try:
            with torch.no_grad():
                return super().predict(utterance, context)
        except RuntimeError as e:
            if str(e).startswith("No dimension to distribute"):
                logger.warning(str(e))
                return []
            else:
                raise e