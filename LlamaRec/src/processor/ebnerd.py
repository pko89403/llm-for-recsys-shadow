from typing import Dict, Any, List, Tuple, Union
import random
from functools import partial
import numpy as np
from langchain_core.prompts import (
    PromptTemplate,
    PipelinePromptTemplate
)

from .util import flatten_with_random_choice
from .base import BaseProcessor

from src.prompt.completion import create_llamarec_completion_chain
from src.models.llamarec.verb import int_to_letter



def convert_to_simple_format(zipped_dataset, user2impr,  article_id2i) -> List[int]:
    """Get articles and convert it to user_history and label data format.
    
    Timeline of the data looks as follows.
        old_article_ids > new_article_ids > clicked_article_ids

    This function just construct one data example for one impression. 

    Args:
        zipped_dataset: zipped dataset of the impression data.
        user2impr: user2impr[user_id] = [impression_i1, impression_i2, ...]
        article_id2i: article_id2i[article_id] = article_index
    
    Returns:
        [user_history_ids, label_id]
    """
    impression_id_list = []
    long_term_history_list = []
    long_term_scroll_list = []
    long_term_read_time_list = []
    short_term_history_list = []
    candidate_pool_list = []
    label_list = []

    for impr_i, impr_id, user_id, inview_ids, fixed_articles, scroll_percents, read_times, clicked_articles, clicked in zipped_dataset:
        ## user_history
        # old_aritcles
        old_article_i_list = list(map(lambda x: article_id2i[x], fixed_articles))
        
        # new_articles
        pred_i = user2impr[user_id].index(impr_i) # fixed 이후에 발생한 impression의 index.
        if clicked_articles:
            new_article_i_list = list(map(lambda x: article_id2i[x], flatten_with_random_choice(clicked_articles[:pred_i])))
        else:
            new_article_i_list = []
        
        ## candidate pool -> impression's article_ids_inview
        random.shuffle(inview_ids) 
        inview_i_list = list(map(lambda x: article_id2i[x], inview_ids))

        # label_list
        if clicked:
            clicked_i_list = list(map(lambda x: article_id2i[x], clicked))

            for label in clicked_i_list:
                impression_id_list.append(impr_id)
                long_term_history_list.append(old_article_i_list)
                long_term_scroll_list.append(scroll_percents)
                long_term_read_time_list.append(read_times)
                short_term_history_list.append(new_article_i_list)
                candidate_pool_list.append(inview_i_list)
                label_list.append(label)
        else:
            impression_id_list.append(impr_id)
            long_term_history_list.append(old_article_i_list)
            long_term_scroll_list.append(scroll_percents)
            long_term_read_time_list.append(read_times)
            short_term_history_list.append(new_article_i_list)
            candidate_pool_list.append(inview_i_list)
            label_list.append("")

    return (
        impression_id_list,
        long_term_history_list,
        long_term_scroll_list,
        long_term_read_time_list,
        short_term_history_list,
        candidate_pool_list,
        label_list
    )
    
    
class LlamaRecProcessorRS(BaseProcessor):
    def __init__(self, ds_articles, user2impr, ds_embeddings=None, seed:int=42):
        super(LlamaRecProcessorRS, self).__init__(ds_articles, ds_embeddings)
        self.article_id2i = {a: i for i, a in enumerate(ds_articles['article_id'])} # article_id -> ds_article's index
        self.user2impr = user2impr
        self.seed = seed

        random.seed(seed)

        self.convert_fn = partial(
            convert_to_simple_format, 
            user2impr=self.user2impr,
            article_id2i=self.article_id2i
            )

    def __call__(self, examples:Dict[str, Any], indices:List[int], start_index:Union[int, None]=None) -> Dict[str, Any]:
        """This call must be used with ds.map with batched=True and with_indices=True."""
        indices = [start_index + i for i in indices]
        zipped_dataset = zip(
            indices,
            examples['impression_id'],
            examples['user_id'], 
            examples['article_ids_inview'],
            examples['article_id_fixed'],
            examples['scroll_percentage_fixed'],
            examples['read_time_fixed'],
            examples.get('clicked_articles', [None] * len(indices)),  # test에는 없음.
            examples.get('article_ids_clicked', [None] * len(indices)) # test에는 없음.
        )
        
        (
            impression_id_list,
            long_term_history_list,
            long_term_scroll_list,
            long_term_read_time_list,
            short_term_history_list,
            candidate_pool_list,
            label_list
        ) = self.convert_fn(zipped_dataset)

        return {
            "impression_id": impression_id_list,
            "long_term_history": long_term_history_list,
            "long_term_scroll": long_term_scroll_list,
            "long_term_read_time": long_term_read_time_list,
            "short_term_history": short_term_history_list,
            "candidate_pool": candidate_pool_list,
            "label": label_list
        }
    
def split_array_efficient(A, B) -> List[List[int]]:
    if sum(B) != len(A):
        raise ValueError("The sum of lengths in B must exactly match the length of A")
    
    # 누적 합을 계산하여 분할 위치 결정
    split_indices = np.cumsum(B)[:-1]  # 마지막 요소는 제외
    return [list(x) for x in np.split(A, split_indices)]


class LlamaRecProcessorCompletion(BaseProcessor):
    def __init__(
            self,
            prompt_template:str,
            instruction_template:str,
            system_message:str,
            input_template:str,
            tokenizer,
            ds_articles,
            ds_embeddings=None,
            scroll_lpt:float=0.01, 
            read_time_lpt:float=0.01, 
            include_na_scroll:bool=False,
            max_history_len:int=30,
            max_candidate_len:int=250,
            label_prefix:str= " ",
            mode:str="train",
    ):
        super(LlamaRecProcessorCompletion, self).__init__(ds_articles, ds_embeddings)

        self.prompt_template = prompt_template
        self.instruction_template = instruction_template
        self.input_template = input_template
        self.system_message = system_message

        full_prompt = PromptTemplate.from_template(self.prompt_template)
        instruction_prompt = PromptTemplate.from_template(
            self.instruction_template,
            partial_variables={"system_message": self.system_message}
        )
        input_prompt = PromptTemplate.from_template(self.input_template)
        input_prompts = [
            ("instruction", instruction_prompt),
            ("input", input_prompt),
        ]
        self.pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_prompt,
            pipeline_prompts=input_prompts
        )

        self.tokenizer = tokenizer
        self.scroll_lpt = scroll_lpt # lower-bound percentile
        self.read_time_lpt = read_time_lpt
        self.include_na_scroll = include_na_scroll
        self.max_history_len = max_history_len
        self.max_candidate_len = max_candidate_len
        self.label_prefix = label_prefix

        self.chain = create_llamarec_completion_chain(self.pipeline_prompt)
        self.mode = mode

    def __call__(self, examples:Dict[str, Any]) -> Dict[str, Any]:
        """This call must be used with ds.map with batched=True."""
        long_term_history_list:List[List[int]] = examples['long_term_history']
        long_term_scroll_list:List[List[int]] = examples['long_term_scroll']
        long_term_read_time_list:List[List[int]] = examples['long_term_read_time']

        ## preprocess long_term history
        max_lth_len = max(map(len, long_term_history_list)) # fixed

        if self.include_na_scroll:
            long_term_scroll_list = [[x if x else 0 for x in scrolls] for scrolls in long_term_scroll_list]

        padded_lth_arr = np.array([np.pad(item, (0, max_lth_len - len(item)), mode='constant',constant_values=-1)
                   for item in long_term_history_list])
        padded_scroll_arr = np.array([np.pad(item, (0, max_lth_len - len(item)), mode='constant',constant_values=np.nan)
                   for item in long_term_scroll_list], dtype='float')
        padded_read_time_arr = np.array([np.pad(item, (0, max_lth_len - len(item)), mode='constant',constant_values=np.nan)
                   for item in long_term_read_time_list], dtype='float')

        scroll_lower_bounds = np.nanpercentile(padded_scroll_arr, self.scroll_lpt, axis=1)
        read_time_lower_bounds = np.nanpercentile(padded_read_time_arr, self.read_time_lpt, axis=1)

        filter_cond = (padded_scroll_arr > scroll_lower_bounds[:, None]) & (padded_read_time_arr > read_time_lower_bounds[:, None])
        filtered_long_term_history = split_array_efficient(padded_lth_arr[filter_cond], filter_cond.sum(axis=1)) # check

        histories = [[*lth, *sth][-self.max_history_len:] for lth, sth, in zip(filtered_long_term_history, examples['short_term_history'])]

        inputs = []
        labels = []
        instructions = []

        for history, candidate, label in zip(histories, examples['candidate_pool'], examples['label']):
            label_i = -1
                
            # if self.mode == "train" and len(candidate) >= self.max_candidate_len:
            #     candidate = candidate[-self.max_candidate_len:]
            # elif self.mode == "train" and len(candidate) < self.max_candidate_len:
            #     # 필요한 추가 아이템 수 계산
            #     needed = self.max_candidate_len - len(candidate)
            #     available_slots = set(range(1, self.ds_articles.num_rows)) - set(candidate) - set([label])
            #     additional_items = np.random.choice(list(available_slots), needed, replace=False)
            #     candidate.extend(additional_items)
            if self.mode == "train":
                candidate = candidate
            else:
                candidate = candidate

            if label in candidate:
                label_i = candidate.index(label)
            else:
                candidate.pop(0)
                candidate.append(label)
                random.shuffle(candidate)
                label_i = candidate.index(label)

            history_title_llamarec = self.ds_articles[history]['title_llamarec']
            candidate_title_llamarec = self.ds_articles[candidate]['title_llamarec']

            completion = self.chain.invoke({
                'user_history': history_title_llamarec,
                'candidate_pool': candidate_title_llamarec,
            }).to_string()

            labels.append(label_i)
            label = int_to_letter(label_i)
            if self.mode == "train":
                instructions.append(completion+self.label_prefix+label)
            else:
                instructions.append(completion+self.label_prefix)
        
        if self.mode == "validation":
            output = self.tokenizer(
                instructions,
                return_tensors=None
            )

            output['label_indexes'] = labels # A->0, B->1, C->2, D, E 
            return output
        else:
            output = {}
            output['instructions'] = instructions # SFTrainer -> tokenizer  ( str)
            return output
            

