#!/bin/sh

python finetune.py \
    --base_model facebook/opt-125m \
    --data_path ML1M \
    --task_type general \
    --output_dir ./LLM4Rec \
    --batch_size 1 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 0.0003 \
    --cutoff_len 4096 \
    --val_set_size 0 \
    --lora_r 2 \
    --lora_alpha 1 \
    --lora_dropout 0.05 \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --warmup_steps 100
