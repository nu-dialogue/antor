import random
import numpy as np
import torch
from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Set random seed to {seed}")