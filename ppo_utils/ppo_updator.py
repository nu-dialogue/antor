import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import NamedTuple, List
import numpy as np

from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

from ppo_utils.core import (
    logprobs_from_logits,
    entropy_from_logits,
    whiten,
    stack_dicts,
    stats_to_np,
    clip_by_value,
    flatten_dict
)
import time

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

class Rollouts:
    class MiniBatch(NamedTuple):
        indices: List[int]
        logprobs: torch.FloatTensor
        values: torch.FloatTensor
        returns: torch.FloatTensor
        advantages: torch.LongTensor
        response_ids: torch.LongTensor
        model_input: torch.LongTensor

    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.trajectory_len = 0
        self.model_input = []
        self.query_ids = []
        self.response_ids = []

        self.logprobs = []
        self.values = []
        self.reward = []

        self.ref_logprobs = []
        self.kl = []
        self.rewards = []
        self.non_score_rewards = []
        self.returns = []
        self.advantages = []
    
    def insert_response(self, query_ids, response_ids, device):
        self.trajectory_len += 1
        query_ids = query_ids.to(device)
        response_ids = response_ids.to(device)
        self.query_ids.append(query_ids)
        self.response_ids.append(response_ids)
        self.model_input.append(torch.cat((query_ids, response_ids), axis=1))
    
    def insert_reward(self, reward, device):
        self.reward.append(reward.to(device))
        # assert self.trajectory_len == len(self.reward)

    def _compute_advantages(self, rewards, values, gamma, lam):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = values.size(1)
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        returns = advantages + values
        return returns, advantages

    def _whiten_advantages(self, advantages):
        advantages_ = torch.cat(advantages, dim=-1)
        mean, var = torch.mean(advantages_), torch.var(advantages_)
        whtn_advs = [(advantages[i]- mean) * torch.rsqrt(var + 1e-8) for i in range(len(advantages))]
        return whtn_advs

    def forward_and_compute_returns(self, policy_model, value_model, ref_policy_model, kl_coef, gamma, lam):
        assert self.trajectory_len >= self.batch_size
        assert self.trajectory_len == len(self.model_input)
        assert self.trajectory_len == len(self.response_ids)
        assert self.trajectory_len == len(self.reward)

        for i in range(self.trajectory_len):
            model_input = self.model_input[i]
            response_ids = self.response_ids[i]
            score = self.reward[i]

            gen_len = response_ids.size(1)
            
            with torch.no_grad():
                logits, values, *_ = policy_model(model_input, rl_forward=True)
                if value_model is not None:
                    values = value_model(model_input)
            values = values[:,-gen_len-1:-1]
            logits = logits[:,-gen_len-1:-1,:]
            logprobs = logprobs_from_logits(logits, response_ids)

            with torch.no_grad():
                ref_logits, _ref_values, *_ = ref_policy_model(model_input, rl_forward=True)
            ref_logits = ref_logits[:,-gen_len-1:-1,:]
            ref_logprobs = logprobs_from_logits(ref_logits, response_ids)

            kl = logprobs - ref_logprobs
            non_score_rewards = -kl_coef * kl
            rewards = non_score_rewards.clone()
            rewards[:, -1] += score

            returns, advantages = self._compute_advantages(rewards=rewards, values=values,
                                                           gamma=gamma, lam=lam)

            self.logprobs.append(logprobs)
            self.ref_logprobs.append(ref_logprobs)
            self.values.append(values)
            self.kl.append(kl) # Size([batch_size(=1)])
            self.rewards.append(rewards)
            self.non_score_rewards.append(non_score_rewards)
            self.returns.append(returns)
            self.advantages.append(advantages)

        self.advantages = self._whiten_advantages(self.advantages)

    def minibatch_iterator(self, minibatch_size):
        if minibatch_size > 1:
            raise NotImplementedError
        sampler = BatchSampler(SubsetRandomSampler(range(self.batch_size)),
                               minibatch_size, drop_last=True)
        for indices in sampler:
            i = indices[0]
            yield Rollouts.MiniBatch(indices=indices, logprobs=self.logprobs[i], values=self.values[i],
                                     returns=self.returns[i], advantages=self.advantages[i],
                                     response_ids=self.response_ids[i], model_input=self.model_input[i])
    
class PPOUpdator:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """
    def __init__(self, policy_model, value_model, ref_policy_model, total_iterations, ppo_config):
        """
        Initialize PPOTrainer.

        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000

        """
        self.ppo_params = ppo_config

        self.policy_model = policy_model
        self.value_model = value_model
        self.ref_policy_model = ref_policy_model
        
        self.policy_optimizer = Adam(policy_model.parameters(), lr=self.ppo_params['lr'])
        if self.value_model is not None:
            logger.info("Train with separated value function")
            self.value_optimizer = Adam(value_model.parameters(), lr=self.ppo_params['lr'])

        total_steps = total_iterations*ppo_config["ppo_epochs"]*(ppo_config["batch_size"]/ppo_config["minibatch_size"])
        if ppo_config["lr_linear_decay"]:
            self.scheduler = LambdaLR(optimizer=self.policy_optimizer,
                                      lr_lambda=lambda steps: 1-(steps/total_steps))
        else:
            self.scheduler = None

        if self.ppo_params['target_kl'] is not None:
            self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                               self.ppo_params['target_kl'],
                                               self.ppo_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])

    def step(self, rollouts: Rollouts, update_vf_only=False):
        """
        Run a PPO optimisation step.

        args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the encoded responses, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """

        timing = dict()
        t0 = time.time()

        t = time.time()
        rollouts.forward_and_compute_returns(policy_model=self.policy_model, value_model=self.value_model, ref_policy_model=self.ref_policy_model,
                                             kl_coef=self.kl_ctl.value, gamma=self.ppo_params["gamma"], lam=self.ppo_params["lam"])
        timing['time/ppo/forward_pass'] = time.time()-t

        t = time.time()
        all_stats = []
        for _ in range(self.ppo_params['ppo_epochs']):
            minibatch_iter = rollouts.minibatch_iterator(minibatch_size=1)
            for mbatch in minibatch_iter:
                if not update_vf_only:
                    train_stats = self.train_minibatch(indices=mbatch.indices, logprobs=mbatch.logprobs, values=mbatch.values,
                                                       returns=mbatch.returns, advantages=mbatch.advantages,
                                                       response_ids=mbatch.response_ids, model_input=mbatch.model_input)
                else:
                    train_stats = self.train_minibatch_vf_only(indices=mbatch.indices, values=mbatch.values, returns=mbatch.returns,
                                                               response_ids=mbatch.response_ids, model_input=mbatch.model_input)
                all_stats.append(train_stats)
        timing['time/ppo/optimize_step'] = time.time()-t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        try:
            train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
            train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)
        except KeyError:
            pass
        if self.scheduler is not None:
            train_stats['policy/lr'] = torch.Tensor(self.scheduler.get_last_lr())

        stats = self.record_step_stats(logprobs=rollouts.logprobs, ref_logprobs=rollouts.ref_logprobs,
                                       non_score_reward=rollouts.non_score_rewards, train_stats=train_stats,
                                       kl_coef=self.kl_ctl.value, kl=rollouts.kl)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time()-t0
        stats.update(timing)
        return stats

    def train_minibatch(self, indices, logprobs, values, returns, advantages, response_ids, model_input):
        """Train one PPO minibatch"""
        pg_loss, vf_loss, train_stats = self.loss(logprobs, values, returns, advantages, response_ids, model_input)
        if self.value_model is not None:
            self.policy_optimizer.zero_grad()
            pg_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            vf_loss.backward()
            self.value_optimizer.step()

        else:
            self.policy_optimizer.zero_grad()
            (pg_loss + vf_loss).backward()
            self.policy_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return train_stats

    def train_minibatch_vf_only(self, indices, values, returns, response_ids, model_input):
        assert self.value_model is not None

        gen_len = response_ids.shape[1]
        vpred = self.value_model(model_input)
        vpred = vpred[:,-gen_len-1:-1]
        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])
        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        self.value_optimizer.zero_grad()
        vf_loss.backward()
        self.value_optimizer.step()
        
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            # loss=dict(policy=None, value=vf_loss.detach(), total=None),
            loss=dict(value=vf_loss.detach()),
            # policy=dict(entropy=None, approxkl=None, policykl=None,
            #             clipfrac=None, advantages=None,
            #             advantages_mean=None, ratio=None),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(vpred=torch.mean(vpred).detach(), error=torch.mean((vpred - returns) ** 2).detach(),
                     clipfrac=vf_clipfrac.detach(), mean=value_mean.detach(), var=value_var.detach()),
        )
        return flatten_dict(stats)

    def loss(self, old_logprobs, values, returns, advantages, response_ids, model_input):
        """Calculate policy and value losses."""
        gen_len = response_ids.shape[1]

        logits, vpred, *_ = self.policy_model(model_input, rl_forward=True)
        if self.value_model is not None:
            vpred = self.value_model(model_input)

        #only the generation part of the values/logprobs is needed
        logits = logits[:,-gen_len-1:-1,:]
        logprobs = logprobs_from_logits(logits, response_ids)
        vpred = vpred[:,-gen_len-1:-1]

        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprobs - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.ppo_params['vf_coef']*vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprobs - old_logprobs)**2)
        policykl = torch.mean(logprobs - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(entropy=entropy.detach(), approxkl=approxkl.detach(), policykl=policykl.detach(),
                        clipfrac=pg_clipfrac.detach(), advantages=advantages.detach(),
                        advantages_mean=torch.mean(advantages).detach(), ratio=ratio.detach()),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(vpred=torch.mean(vpred).detach(), error=torch.mean((vpred - returns) ** 2).detach(),
                     clipfrac=vf_clipfrac.detach(), mean=value_mean.detach(), var=value_var.detach()),
        )
        return pg_loss, self.ppo_params['vf_coef']*vf_loss, flatten_dict(stats)


    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl_dist = torch.cat(data['kl'],dim=-1)
        logprobs_dist = torch.cat(data['logprobs'], dim=-1)
        ref_logprobs_dist = torch.cat(data['ref_logprobs'], dim=-1)

        mean_kl = torch.cat( [kl.sum(dim=-1) for kl in data['kl']] ).mean()
        mean_entropy = torch.cat( [-logprobs.sum(dim=-1) for logprobs in data['logprobs']] ).mean()

        mean_non_score_reward = torch.cat([non_score_reward.sum(dim=-1) for non_score_reward in data['non_score_reward']] ).mean()

        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl_dist,
            'objective/logprobs': logprobs_dist,
            'objective/ref_logprobs': ref_logprobs_dist,
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats