from pathlib import Path
from datasets import load_dataset
from datasets import load_from_disk, concatenate_datasets
import os 

from src.processor import util
from src.processor.ebnerd import LlamaRecProcessorCompletion

# PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n ### Input:\n{input}\n\n ### Response:\n"
PROMPT_TEMPLATE = "### Instruction: {instruction}\n Input: {input}\n ### Response:"

INSTRUCTION_TEMPLATE = """{system_message}"""
SYSTEM_TEMPLATE = """Given user history in chronological order, recommend an item from the candidate pool with its index letter."""
INPUT_TEMPLATE = """User history: {user_history} \n Candidate pool: {candidate_pool}"""
    
# NUM_SHARDS = 10000

def get_dataset(tokenizer, path="mbhr/ebnerd-llama-rec", data_dir="300k", split="train", max_history_len=30, max_candidate_len=250, mode="train"):
    print(f"get_dataset {path}, {data_dir}, {split}")
#     ds = load_dataset(path, data_dir, split=split)
    
#     ds_articles = load_dataset("mbhr/EB-NERD", "articles", split='large')
#     ds_articles = ds_articles.map(util.add_title_llamarec_short, batched=True)
    
#     processor = LlamaRecProcessorCompletion(
#         prompt_template=PROMPT_TEMPLATE,
#         instruction_template=INSTRUCTION_TEMPLATE,
#         system_message=SYSTEM_TEMPLATE,
#         input_template=INPUT_TEMPLATE,
#         tokenizer=tokenizer,
#         ds_articles=ds_articles,
#         max_history_len=max_history_len,
#         max_candidate_len=max_candidate_len,
#         label_prefix="\n",
#         mode=mode,
#     )
    
    num_shards=10
    if split == "validation":
        num_shards=1

    # for i in range(num_shards):
    #     sub_ds = ds.shard(num_shards=num_shards, index=i, contiguous=True)
    #     new_ds = sub_ds.map(
    #         processor,
    #         batched=True,
    #         remove_columns=sub_ds.column_names,
    #         load_from_cache_file=False,
    #         num_proc=4,
    #     )
    #     new_ds.save_to_disk(os.path.join('/data02', 'preprocessed', 'llama_rec_free', split, f"{i}"))
        
    ## load and concatenate
    new_ds = concatenate_datasets([load_from_disk(
        os.path.join('/data02', 'preprocessed', 'llama_rec_free', split, f"{i}")
        ) for i in range(num_shards)])
    
    # new_ds = new_ds.shuffle()
    return new_ds