# Models
# Mabeck/Heidrun-Mistral-7B-chat

from typing import List, Dict, Any, Tuple
from operator import itemgetter

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate       
)
from langchain_core.runnables import RunnableLambda
from langchain.schema import HumanMessage, SystemMessage

from llmebr.models.llamarec.verb import int_to_letter


def _format_user_history(user_history:List[str]) -> str:
    return '\n'.join([f"({i}) {x}"for i, x in enumerate(user_history)])

def _format_condidate_pool(candidate_pool:List[str]) -> str:
    return '\n'.join([f"({int_to_letter(i)}) {x}"for i, x in enumerate(candidate_pool)])

def _format_label(label_i:int) -> str:
    return chr(ord('A')+label_i)

def output_parser(prompt_value):
    return prompt_value
## sample


def create_llamarec_completion_chain(pipeline_prompt):
    """
    """
    return (
        {
            "user_history": itemgetter("user_history") | RunnableLambda(_format_user_history),
            "candidate_pool": itemgetter("candidate_pool") | RunnableLambda(_format_condidate_pool),
        }
        | pipeline_prompt
        | output_parser
    )
