import re
import string
from typing import Any

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    
    def remote_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remote_punc(lower(s))))

def EM(answer: str, key: str) -> bool:
    """정규화된 답변과 정규화된 키가 일치하는지 확인합니다.

    Args:
        answer (str): 확인할 답변입니다.
        key (str): 비교할 키입니다.

    Returns:
        bool: 정규화된 답변과 정규화된 키가 일치하면 True, 그렇지 않으면 False입니다.
    """
    return normalize_answer(answer) == normalize_answer(key)

def is_correct_qa(answer: str, gt_answer: str) -> bool:
    if isinstance(answer, str):
        return EM(answer, gt_answer)
    else:
        return EM(str(answer), gt_answer)
    
def is_correct_rp(answer: float, gt_answer: float) -> bool:
    return answer == gt_answer

def is_correct_sr(answer: list[int], gt_answer: int) -> bool:
    if len(answer) == 0:
        return False
    return answer[0] == gt_answer

def is_correct(task: str, answer: Any, gt_answer: Any) -> bool:
    """주어진 task에 따라 정답과 비교하여 올바른지 확인합니다.

    Args:
        task (str): 작업 유형입니다.
        answer (Any): 확인할 답변입니다.
        gt_answer (Any): 비교할 정답입니다.

    Raises:
        ValueError: 지원되지 않는 작업 유형일 경우 발생합니다.

    Returns:
        bool: 정답과 비교하여 올바른 경우 True를 반환하고, 그렇지 않으면 False를 반환합니다.
    """
    if task == "qa":
        return is_correct_qa(answer, gt_answer)
    elif task == "rp":
        return is_correct_rp(answer, gt_answer)
    elif task == "sr":
        return is_correct_sr(answer, gt_answer)
    else:
        raise ValueError(f"Unsupported task type: {task}")
