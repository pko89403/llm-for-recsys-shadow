import os
import sys
from typing import List, Optional

from dataclasses import dataclass, field
from click import Option
from transformers import HfArgumentParser, TrainingArguments, set_seed
from e4srec.utils.prompter import Prompter

@dataclass
class ModelArguments:
    base_model: Optional[str] = field(
        default="garage-bAInd/Platypus2-70B-instruct",
        metadata={"help": "The base model to use for finetuning"}
    )
    data_path: Optional[str] = field(
        default="ML1M",
        metadata={"help": "The path to the data to use for finetuning"}
    )
    cache_dir: Optional[str] = field(
        default="",
        metadata={"help": "The cache directory to save the model"}
    )
    task_type: Optional[str] = field(
        default="general",
        metadata={"help": "The task type of the model"}
    )
    batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "The batch size for training"}
    )
    micro_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The micro batch size for training"}
    )
    num_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of epochs for training"}
    )
    cutoff_len: int = field(
        default=4096,
        metadata={"help": "The cutoff length for training"}
    )
    val_set_size: int = field(
        default=0,
        metadata={"help": "The size of the validation set"}
    )
    lr_scheduler: Optional[str] = field(
        default="cosine",
        metadata={"help": "The learning rate scheduler"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "The number of local attention heads"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha for local attention"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "The dropout rate for local attention"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["gate_proj", "down_proj", "up_proj"],
        metadata={"help": "The target modules for local attention"}
    )
    train_on_inputs: bool = field(
        default=False,
        metadata={"help": "Whether to train on inputs"}
    )
    add_eos_token: bool = field(    
        default=False, # if False, masks out inputs in loss
        metadata={"help": "Whether to add an EOS token"}
    )
    wandb_project: Optional[str] = field(
        default="",
        metadata={"help": "The wandb project"}
    )
    wandb_run_name: Optional[str] = field(
        default="",
        metadata={"help": "The wandb run name"}
    )
    wandb_watch: Optional[str] = field(
        default="",
        metadata={"help": "The wandb watch"}
    )
    wandb_log_model: Optional[str] = field(
        default="",
        metadata={"help": "The wandb log model"}
    )
    prompt_template_name: Optional[str] = field(
        default="alpaca",
        metadata={"help": "The prompt template name"}
    )


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
        print((model_args, training_args))

    gradient_accumulation_steps = model_args.batch_size // model_args.micro_batch_size
    prompter = Prompter()
    
        
    train(model_args, training_args)
