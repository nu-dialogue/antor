import random
import math
from collections import Counter

from common_utils.multiwoz_data import MultiWOZData
from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

class PPOTrainData:
    def __init__(self, multiwoz_data: MultiWOZData, parts_used, batch_size, shuffle, infinite):
        self.parts = parts_used
        act_resp_data = multiwoz_data.get_act_resp_data(parts=self.parts)
        self.data = []
        for part_data in act_resp_data.values():
            self.data += part_data

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.infinite = infinite
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        count = 1
        while count > 0 :
            if not self.infinite:
                count -= 1
            if self.shuffle:
                self.data = random.sample(self.data, len(self.data))
            for i in range(0, len(self.data), self.batch_size):
                if i+self.batch_size < len(self.data):
                    yield self.data[i:i+self.batch_size]
                else:
                    break

    def sample_batch(self):
        return random.sample(self.data, k=self.batch_size)

class ActionIDF:
    def __init__(self, multiwoz_data: MultiWOZData, parts_used, include_value=False) -> None:
        self.include_value = include_value
        act_resp_data = multiwoz_data.get_act_resp_data(parts=parts_used)
        num_turns = sum([len(part_data) for part_data in act_resp_data.values()])
        actions = []
        for part_data in act_resp_data.values():
            for turn in part_data:
                for act in turn["system_action"]:
                    if self.include_value:
                        actions.append(self.flatten_action(act))
                    else:
                        actions.append(self.flatten_action(act[:-1]))
        self.action_idf = self._compute_idf(doc_len=num_turns, actions=actions)

    def __getitem__(self, action: list):
        assert isinstance(action, list) and len(action) == 4
        if self.include_value:
            action = self.flatten_action(action)
        else:
            action = self.flatten_action(action[:-1])
        if not action in self.action_idf:
            logger.warning(f"{action} is unknown action.")
            return 1.0
        else:
            return self.action_idf[action]

    def flatten_action(self, action):
        return "-".join(action)

    def _compute_idf(self, doc_len, actions):
        action_idf = {}
        action_freq = Counter(actions).most_common()
        for act, cnt in action_freq:
            action_idf[act] = math.log(doc_len/cnt)
        return action_idf