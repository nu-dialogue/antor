import os
import sys
import json
from transformers import GPT2Tokenizer, GPT2PreTrainedModel, GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from torch import nn

from logging import getLogger
from common_utils.log import set_logger
logger = getLogger(__name__)
set_logger(logger)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self, config):
        super().__init__()
        self.detach_head = False
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = nn.Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = nn.Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()

        self.first_dropout = nn.Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output

class GPT2HeadWithValueHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.v_head = ValueHead(config)

        self.post_init()

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rl_forward=False
    ):
        assert labels is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        lm_logits = self.lm_head(hidden_states)
        
        loss = None

        if not rl_forward:
            if not return_dict:
                return (lm_logits,) + transformer_outputs[1:]

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )

        else:
            value = self.v_head(hidden_states).squeeze(-1)
            return (lm_logits, value) + transformer_outputs[1:]

    def save_checkpoint(self, tokenizer, output_dpath, prefix, eval_results=None):
        output_dpath = os.path.join(output_dpath, prefix)
        os.makedirs(output_dpath, exist_ok=True)
        logger.info(f'Saving model checkpoint to {output_dpath}')
        self.save_pretrained(output_dpath)
        tokenizer.save_pretrained(output_dpath)
        if eval_results is not None:
            result_fpath = os.path.join(output_dpath, "eval_results.json")
            json.dump(eval_results, open(result_fpath, "w"), indent=4)

class GPT2ValueHeadModel(GPT2PreTrainedModel):
    """
    From transformers.GPT2ForTokenClassification
    """
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.v_head = ValueHead(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        value = self.v_head(hidden_states) # size([batch_size, seq_len, 1])
        return value.view(1, -1)

def build_gpt2(gpt2_config):
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_config["tokenizer_name"])

    policy_gpt2 = GPT2HeadWithValueHeadModel.from_pretrained(gpt2_config["pretrained_model_dpath"])
    ref_policy_gpt2 = GPT2HeadWithValueHeadModel.from_pretrained(gpt2_config["ref_model_dpath"])
    _ = policy_gpt2.to(DEVICE)
    _ = ref_policy_gpt2.to(DEVICE)
    
    if gpt2_config["separate_vf"]:
        value_gpt2 = GPT2ValueHeadModel.from_pretrained(gpt2_config["pretrained_model_dpath"])
        _ = value_gpt2.to(DEVICE)
    else:
        value_gpt2 = None

    return tokenizer, policy_gpt2, value_gpt2, ref_policy_gpt2