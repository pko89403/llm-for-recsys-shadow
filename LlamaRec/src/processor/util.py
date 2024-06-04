from typing import List

import random

def flatten_with_random_choice(A: List[List[int]]) -> List[int]:
    """Flatten a list of list of integers with random choice
    
    Args:
        A (List[List[int]]): List of list of integers
    
    Returns:
        List[int]: Flattened list of integers
    """
    result = []
    for sublist in A:
        if sublist:
            selected_item = random.choice(sublist)
            result.append(selected_item)
    return result

def add_title_llamarec(examples:Dict[str, Any]):
    """LlamaRec style article title
    pattern: {title} | {subtitle} (yyyy-mm-dd hh:mm:ss)
    """
    title_llamarec_list = [f"{title.replace('|', ' ')} | {subtitle.replace('|', ' ')} ({format_datetime_danish(dt)})" 
     for title, subtitle, dt in zip(examples['title'], examples['subtitle'], examples['published_time'])]

    return {
        "title_llamarec" : title_llamarec_list
    }

def add_title_llamarec_short(examples:Dict[str, Any]):
    """LlamaRec style article title
    pattern: {title} | (yyyy-mm-dd hh:mm:ss)
    """
    title_llamarec_list = [f"{title.replace('|', ' ')} ({format_datetime_danish(dt)})" 
     for title, dt in zip(examples['title'], examples['published_time'])]

    return {
        "title_llamarec" : title_llamarec_list
    }