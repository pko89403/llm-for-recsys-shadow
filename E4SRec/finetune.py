import os
import sys
from typing import List, Optional
import pickle
from dataclasses import dataclass, field

import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments, set_seed
from loguru import logger

from e4srec.model import LLM4Rec
from e4srec.dataset import BipartiteGraphCollator, BipartiteGraphDataset
from e4srec.utils.config import ModelArguments
from e4srec.utils.prompter import Prompter


def train(
    model_args: ModelArguments,
    training_args: TrainingArguments,
):
    print("output")
    print(model_args)
    print(training_args)
    # Train the model
    # ...
    pass

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 기본값을 설정하는 코드 추가
        if not any("--output_dir" in arg for arg in sys.argv):
            sys.argv += ["--output_dir", "outputs"]
        model_args, training_args = parser.parse_args_into_dataclasses()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        pass

    gradient_accumulation_steps = model_args.batch_size // model_args.micro_batch_size
    prompter = Prompter(model_args.prompt_template_name)
    
    device_map = "cpu"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        graident_accumulation_steps = gradient_accumulation_steps // world_size
        logger.info("gradient_accumulation_steps: ", gradient_accumulation_steps)
    
    # Check if parameter passed or if set within environ
    use_wandb = len(model_args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(model_args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project
    if len(model_args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = model_args.wandb_watch
    if len(model_args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = model_args.wandb_log_model

    logger.info(f"Task Type : {model_args.task_type}")
    if model_args.task_type == "general":
        dataset = BipartiteGraphDataset(
            dataset=os.path.join("datasets", model_args.task_type, model_args.data_path)
        )

        user_emb_path = os.path.join("datasets", model_args.task_type, model_args.data_path, "VanillaMF_user_embed.pkl")
        user_embed = pickle.load(open(user_emb_path, "rb"))
        
        item_emb_path = os.path.join("datasets", model_args.task_type, model_args.data_path, "VanillaMF_item_embed.pkl")
        item_embed = pickle.load(open(item_emb_path, "rb")) # torch.Size([3043, 64])
        item_embed = torch.cat([item_embed.mean(dim=0).unsqueeze(0), item_embed], dim=0) # torch.Size([3044, 64]) COLD START ITEM
        
        data_collator = BipartiteGraphCollator()
        
        logger.info(f"Total User Count : {dataset.n_user}, Total Item Count : {dataset.m_item}")
        logger.info(f"user_embed shape : {user_embed.shape}, item_embed shape : {item_embed.shape}")
        logger.info(f"Example Train Data Sample at index 0 : {dataset[0]}")
    elif model_args.task_type == "sequential":
        pass

    model = LLM4Rec(
        base_model=model_args.base_model,
        task_type=model_args.task_type,
        cache_dir=model_args.cache_dir,
        input_dim=64,
        output_dim=dataset.m_item,
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        lora_target_modules=model_args.lora_target_modules,
        device_map=device_map,
        instruction_text=prompter.generate_prompt(model_args.task_type),
        user_embeds=user_embed,
        input_embeds=item_embed,
    )
    logger.info(model)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=model_args.micro_batch_size,
            use_cpu=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=training_args.warmup_steps,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            fp16=False,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps" if model_args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if model_args.val_set_size > 0 else None,
            save_steps=1000,
            lr_scheduler_type=model_args.lr_scheduler,
            output_dir=training_args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if model_args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=training_args.group_by_length,
            report_to="none",
            run_name=None,
        ),
        data_collator=data_collator,
    )
    
    trainer.train()
