#!/usr/bin/env python
# coding: utf-8

import os 
# os.environ["HF_DATASETS_CACHE"] = "/data02/hf_datasets_cache"
from typing import *
from functools import partial

import numpy as np
import datasets
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from trl import SFTTrainer
from trl.trainer import SFTConfig
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)
from transformers.trainer import is_datasets_available
from transformers import DataCollatorForLanguageModeling
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from accelerate import Accelerator
from accelerate import PartialState

from llmebr.models.llamarec.model import LlamaRec, get_llamarec_tokenizer
from llmebr.models.llamarec.dataset import get_dataset
from llmebr.models.llamarec.verb import int_to_letter, ManualVerbalizer
import warnings


def get_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = LlamaRec.from_pretrained(
        MODEL_BASE,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map = {'': PartialState().local_process_index}
    )
    
    model.gradient_checkpointing_enable()
    
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, peft_config

def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks):
    metrics = {}
    labels = F.one_hot(labels, num_classes=scores.size(1)) # A->1 B->2  F.one_hot(1, 250)
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(
                labels.device), labels.sum(1).float())).mean().cpu().item()
        
        metrics['MRR@%d' % k] = \
            (hits / torch.arange(1, k+1).unsqueeze(0).to(
                labels.device)).sum(1).mean().cpu().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                             for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()


    return metrics


def compute_metrics_for_ks(ks, verbalizer):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels).view(-1)
        scores = verbalizer.process_logits(logits) # 토큰에 대한 확률값들이 (250 Tokens )
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels, ks)
        return metrics

    return compute_metrics

def compute_metrics_for_ks_wo_verb(ks):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits).clone().detach().requires_grad_(False).to('cpu')
        labels = torch.tensor(labels).view(-1).clone().detach().requires_grad_(False).to('cpu')
        metrics = absolute_recall_mrr_ndcg_for_ks(logits, labels, ks)
        return metrics

    return compute_metrics

def preprocess_logits_for_metrics_for_vn(verbalizer):
    def preprocess_logits_for_metrics(logits, labels):
        logits = torch.tensor(logits).clone().detach().requires_grad_(False).to(labels.device)
        pred_ids = verbalizer.process_logits(logits) # 토큰에 대한 확률값들이 (250 Tokens )

        return pred_ids

    return preprocess_logits_for_metrics

    """
    Original Trainer may have a memory leak.
    This is a workaround to avioit storing too many tensors are not needed
    """


def llama_collate_fn_w_truncation(llm_max_length):
    def llama_collate_fn(batch):
        """
            dataloader -> batch_data 사이즈가 같아야되자나?
            
            batch_size가 10가 있으면 -> 10개 토큰 길이 중 최대를 패딩을 만들어주더라고?
        """
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        example_max_length = max([len(batch[idx]['input_ids']) for idx in range(len(batch))])
        max_length = min(llm_max_length, example_max_length)
        
        for i in range(len(batch)):
            input_ids = batch[i]['input_ids']
            attention_mask = batch[i]['attention_mask']
            labels = batch[i]['label_indexes'] # A, B, C, D, E
            
            if len(input_ids) >= max_length:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]
            elif len(input_ids) < max_length:
                padding_length = max_length - len(input_ids)
                input_ids = input_ids + [12801] * padding_length # 12801 eot token ( llama3 )
                attention_mask = attention_mask + [0] * padding_length # 1, 0(무시)
            
            all_input_ids.append(torch.tensor(input_ids).long())
            all_attention_mask.append(torch.tensor(attention_mask).long())
            all_labels.append(torch.tensor(labels).long())
        
        return {
            'input_ids': torch.vstack(all_input_ids),
            'attention_mask': torch.vstack(all_attention_mask),
            'labels': torch.vstack(all_labels)
        }
    return llama_collate_fn


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, verbalizer_collator=None, **kwargs):
        super().__init__(*args, **kwargs, model_init_kwargs=None)
        self.verbalizer_collator = verbalizer_collator
        
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.verbalizer_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return self.accelerator.prepare(eval_dataloader)

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling): # trl collator
    """
       tensor(str:instructions) -> BOOM -> max -> padding 
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """
    
    # 250 -> Error O -> Tokenize
    # 10 -> Error X

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        for example in examples:
            del example['instructions']

        # print("before super torch_call !!!!!")
        # print(examples)
        batch = super().torch_call(examples)
        # print("after super torch call !!!!!")
        # print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['labels'].shape)
        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch

if __name__ =="__main__":
    wandb.init(
        project="recsys2024",
        group="DeepSpeed",
        name="llamarec-llama3-8b-instruct",
        tags=["LlamaRec", "llama3-8b-hf"],
        config={
            "log_model": "checkpoint",
        }
    )
    
    MODEL_BASE = "meta-llama/Meta-Llama-3-8B"
    MAX_HISTORY_LEN = 30 
    MAX_CANDIDATE_LEN = 250 # 유동적으로
    MAX_SEQ_LEN = 4000 # 다시 구하기 
    
    model, peft_config = get_model()
    tokenizer = get_llamarec_tokenizer(MODEL_BASE)
    verbalizer = ManualVerbalizer(
        tokenizer=tokenizer,
        prefix="",
        multi_token_handler="mean",
        post_log_softmax=False,
        classes=list(range(MAX_CANDIDATE_LEN)),
        label_words={i: int_to_letter(i) for i in range(MAX_CANDIDATE_LEN)},
    )

    train_dataset = get_dataset(tokenizer, "mbhr/ebnerd-llama-rec", data_dir="300k", split="train", max_history_len=MAX_HISTORY_LEN, mode="train")
    eval_dataset = get_dataset(tokenizer, "mbhr/ebnerd-llama-rec", data_dir="300k", split="validation", mode="validation", max_history_len=MAX_HISTORY_LEN)
    
    print("########################")
    print("Done Getting Dataset ...")
    print("########################")
        
    response_template_string = " ### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template_string, tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir="llama3-8-ebnerd",
        dataset_batch_size=1000,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        eval_accumulation_steps=3,
        gradient_accumulation_steps=3,
        num_train_epochs=3,
        optim="paged_adamw_32bit",
        logging_strategy="steps",
        logging_steps=300,
        eval_strategy="steps",
        eval_steps=3000,
        save_strategy="steps",
        save_steps=3000,
        learning_rate=1e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="wandb",
        save_total_limit=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        disable_tqdm=False, # disable tqdm since with packing values are in correct
        remove_unused_columns=False, # True
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="instructions",
    )
    
    trainer = CustomSFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        verbalizer_collator=llama_collate_fn_w_truncation(MAX_SEQ_LEN), # VALIDATION COLLATOR
        compute_metrics=compute_metrics_for_ks_wo_verb([1,5,10]),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics_for_vn(verbalizer),
        peft_config=peft_config,
    )
    torch.cuda.empty_cache()

    trainer.train()

