import torch
import torch.nn as nn
from tqdm import tqdm

from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

from sys_nlg.gpt2.sl_dataset import prepare_act_resp_datasets, get_dataloader
from sys_nlg.gpt2.utils import get_optimizer_scheduler

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SupervisedTrainer:
    def __init__(self, model, tokenizer, multiwoz_data, sl_config, lm_task_type, act_bos_token, resp_bos_token) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.multiwoz_data = multiwoz_data
        self.lm_task_type = lm_task_type

        self.pad_token_id = tokenizer.eos_token_id
        self.act_bos_token = act_bos_token
        self.resp_bos_token = resp_bos_token
        self.ignore_index = -1

        self.max_grad_norm = sl_config['max_grad_norm']

        self.batch_size = sl_config['batch_size']
        self.eval_batch_size = sl_config['eval_batch_size']
        self.epoch_num = sl_config['epoch_num']
        self.gradient_accumulation_steps = sl_config['gradient_accumulation_steps']
        self.report_interval = sl_config['report_interval']

        self.train_size_ratio = sl_config['train_size_ratio']

        # optim
        self.learning_rate = sl_config['learning_rate']
        self.adam_epsilon = sl_config['adam_epsilon']
        self.weight_decay = sl_config['weight_decay']
        self.warmup_steps = sl_config['warmup_steps']

        # self.earlystopping_patience = sl_config['earlystoping_patience']
        self.checkpoints_output_dpath = sl_config['checkpoints_output_dpath']

        self.best_checkpoint_fpath = None
        self.tb_writer = None

    def load_dataset(self):
        datasets = prepare_act_resp_datasets(multiwoz_data=self.multiwoz_data,
                                             tokenizer=self.tokenizer,
                                             act_bos_token=self.act_bos_token,
                                             resp_bos_token=self.resp_bos_token,
                                             train_size_ratio=self.train_size_ratio)
        self.train_dataset = datasets["train"]
        self.valid_dataset = datasets["val"]

    def _compute_loss_and_accuracy(self, lm_logits, labels, resp_masks):
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_resp_masks = resp_masks.to(torch.bool)[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_resp_masks = shift_resp_masks.view(-1)

        pred_ids = shift_logits.max(1)[1]
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')
        loss = criterion(shift_logits, shift_labels) # Averaged over the num_targets in the criterion()

        not_ignore_masks = shift_labels.ne(self.ignore_index)
        num_corrects = pred_ids.eq(shift_labels).masked_select(not_ignore_masks).sum().item()
        num_outputs = not_ignore_masks.long().sum().item()

        num_resp_corrects = pred_ids.eq(shift_labels).masked_select(shift_resp_masks).sum().item()
        num_resp_outputs = shift_resp_masks.long().sum().item()

        score = {"num_corrects": num_corrects, "num_outputs": num_outputs,
                 "num_resp_corrects": num_resp_corrects, "num_resp_outputs": num_resp_outputs}
        return loss, score

    def train(self):
        num_examples = len(self.train_dataset)
        train_data_size = int(num_examples * self.train_size_ratio)
        total_steps = (train_data_size * self.epoch_num) // (self.gradient_accumulation_steps * self.batch_size)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {self.epoch_num}")
        logger.info(f"  Batch size = {self.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {total_steps}")

        train_iter = get_dataloader(lm_task_type=self.lm_task_type,
                                    dataset=self.train_dataset,
                                    batch_size=self.batch_size,
                                    pad_token_id=self.pad_token_id,
                                    ignore_index=self.ignore_index,
                                    is_train=True)
        optimizer, scheduler = get_optimizer_scheduler(self.model,
                                                       learning_rate=self.learning_rate,
                                                       adam_epsilon=self.adam_epsilon,
                                                       weight_decay=self.weight_decay,
                                                       warmup_steps=self.warmup_steps,
                                                       total_steps=total_steps)

        best_score = {'resp_accuracy': float('-inf')}

        global_step = 0
        # for _ in tqdm(range(self.epoch_num), desc="Epoch"):
        for epoch_id in range(self.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0

            num_corrects = 0
            num_outputs = 0
            num_resp_corrects = 0
            num_resp_outputs = 0
            
            self.model.zero_grad()
            for batch in tqdm(train_iter, desc="Iteration"):
                input_ids = batch.input_ids.to(DEVICE)
                labels = batch.labels.to(DEVICE)
                resp_masks = batch.resp_masks.to(DEVICE)
    
                # loss
                self.model.train()
                outputs = self.model(input_ids)
                loss, score = self._compute_loss_and_accuracy(lm_logits=outputs["logits"],
                                                              labels=labels, resp_masks=resp_masks)
                loss /= self.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                epoch_step += 1
                num_corrects += score["num_corrects"]
                num_outputs += score["num_outputs"]
                num_resp_corrects += score["num_resp_corrects"]
                num_resp_outputs += score["num_resp_outputs"]

                # step, wrt gradient_accumulation_steps, clip grad norm
                if (epoch_step+1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    # global_step: actual step the optimizer took
                    global_step += 1

                    # logging: loss, lr... after certain amount of steps
                    if self.report_interval > 0 and global_step % self.report_interval == 0:
                        loss_scalar = (tr_loss - logging_loss) / self.report_interval
                        logging_loss = tr_loss
                        accuracy = num_corrects / num_outputs
                        resp_accuracy = num_resp_corrects / num_resp_outputs
                        num_corrects = 0
                        num_outputs = 0
                        num_resp_corrects = 0
                        num_resp_outputs = 0
                        logger.info(f'Global step: {global_step}, epoch step: {epoch_step}, ' \
                            + f'interval loss: {loss_scalar:.4f}, interval acc: {accuracy:.4f}, '\
                                + f'interval resp acc: {resp_accuracy:.4f}')
            # validate
            # add to tensorboard...
            eval_results = self.evaluate(epoch_id)

            # save model... 
            if best_score['resp_accuracy'] < eval_results['resp_accuracy']:
                best_score.update(eval_results)
                self.model.save_checkpoint(self.tokenizer, self.checkpoints_output_dpath, f"{self.lm_task_type}.{epoch_id}", eval_results)

    def evaluate(self, epoch_id, prefix=""):
        logger.info(f"***** Running evaluation {prefix} *****")
        logger.info(f"  Num examples = {len(self.valid_dataset)}")
        logger.info(f"  Batch size = {self.eval_batch_size}")

        valid_iter = get_dataloader(lm_task_type=self.lm_task_type,
                                    dataset=self.valid_dataset,
                                    batch_size=self.eval_batch_size,
                                    pad_token_id=self.pad_token_id,
                                    ignore_index=self.ignore_index,
                                    is_train=False)

        eval_loss = 0.0
        nb_eval_steps = 0
        num_corrects = 0
        num_outputs = 0
        num_resp_corrects = 0
        num_resp_outputs = 0
        self.model.eval()
        for batch in tqdm(valid_iter, desc="Evaluating"):
            indices = batch.indices.to(DEVICE)
            input_ids = batch.input_ids.to(DEVICE)
            labels = batch.labels.to(DEVICE)
            resp_masks = batch.resp_masks.to(DEVICE)
            with torch.no_grad():
                outputs = self.model(input_ids)
            loss, score = self._compute_loss_and_accuracy(lm_logits=outputs["logits"],
                                                          labels=labels, resp_masks=resp_masks)
            eval_loss += loss.item()
            num_corrects += score["num_corrects"]
            num_outputs += score["num_outputs"]
            num_resp_corrects += score["num_resp_corrects"]
            num_resp_outputs += score["num_resp_outputs"]
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        results = {"epoch": epoch_id, "loss": eval_loss, "perplexity": perplexity.item(),
                   "accuracy": num_corrects / num_outputs, "resp_accuracy": num_resp_corrects / num_resp_outputs}

        logger.info("***** Eval results {} *****".format(prefix))
        for key, value in results.items():
            logger.info(f"  {key} = {value}")
        return results