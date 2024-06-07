from ast import Not
import re
import json
from typing import Any

from torch import Value

def parse_action(action: str, json_mode: bool = False) -> tuple[str, Any]:
    """Parse action string to action type and argument

    Args:
        action (str): _description_
        json_mode (bool, optional): _description_. Defaults to False.

    Returns:
        tuple[str, Any]: _description_
    """
    if json_mode:
        try:
            json_action = json.loads(action)
            return json_action["type"], json_action["content"]
        except:
            return "Invalid", None
    else:
        pattern = r'^(\w+)\[(.*)\]$' # 하나의 문자열을 받아서, [ ] 안에 있는 문자열을 argument로 받는다.
        match = re.match(pattern, action)

        if match:
            action_type = match.group(1) # match.group(0)은 전체 문자열
            argument = match.group(2) # match.group(1)은 첫번째 괄호 안의 문자열
            return action_type, argument
        else:
            return "Invalid", None

def parse_raw_answer(answer: str, *args, **kwargs) -> dict[str, bool | str]:
    return {
        'valid': True,
        'answer': answer
    }


def parse_rating_answer(answer: str | int | float, json_mode: bool = False, *args, **kwargs) -> dict[str, float | str]:
    """Parse rating answer

    Args:
        answer (str | int | float): _description_
        json_mode (bool, optional): _description_. Defaults to False.

    Returns:
        dict[str, float | str]: _description_
    """
    try:
        answer = float(answer)
        if answer < 1 or answer > 5:
            return {
                "valid": False,
                "answer": "0",
                "message": "Rating should be in range [1, 5]."
            }
    except (ValueError, TypeError):
        return {
            "valid": False,
            "answer": "0",
            "message": "Rating should be a float number."
        }
    except Exception:
        return {
            "valid": False,
            "answer": "0",
            "message": "Other Exception when parsing rating."
        }
    return {
        "valid": True,
        "answer": answer
    }


def parse_ranking_answer(answer: str | Any, gt_answer: int, n_candidate: int, json_mode: bool = False, *args, **kwargs) -> dict[str, bool | list[int]]:
    """Parse ranking answer

    Args:
        answer (str | Any): _description_
        gt_answer (int): _description_
        n_candidate (int): _description_
        json_mode (bool, optional): _description_. Defaults to False.

    Returns:
        dict[str, bool | list[int]]: _description_
    """
    if not json_mode:
        candidates = answer.split(",")
    else:
        if isinstance(answer, list):
            candidates = answer
        elif isinstance(answer, str):
            candidates = answer.split(",")
        else:
            return {
                "valid": False,
                "answer": [],
                "message": "Answer should be a permutated list of candidate ids."
            }

    try:
        length = len(candidates)
    except TypeError:
        return {
            "valid": False,
            "answer": [],
            "message": "Answer should be a permutated list of candidate ids."
        }
    except Exception:
        return {
            "valid": False,
            "answer": [],
            "message": "Other Exception when parsing ranking answer."
        }

    if length != n_candidate:
        return {
            "valid": False,
            "answer": [],
            "message": f"Answer should contain {n_candidate} ids, which is the same as the number of candidates in the question."
        }
    else:
        try:
            answer = [int(c) for c in candidates]
            if gt_answer not in answer:
                return {
                    "valid": False,
                    "answer": [],
                    "message": "Answer should contain all the candidate ids."
                }
        except (ValueError, TypeError):
            return {
                "valid": False,
                "answer": [],
                "message": "The ids in the answer list should be integers."
            }
    return {
        "valid": True,
        "answer": answer
    }


def parse_answer(type: str, *args, **kwargs) -> dict[str, Any]:
    """Parse answer

    Args:
        type (str): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        dict[str, Any]: _description_
    """
    if type == "qa" or type == "chat" or type == "gen":
        return parse_raw_answer(*args, **kwargs)
    elif type == "rp":
        return parse_rating_answer(*args, **kwargs)
    elif type == "sr":
        return parse_ranking_answer(*args, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported task: {type}")


def init_answer(type: str) -> Any:
    if type == "qa" or type == "chat" or type == "gen":
        return ""
    elif type == "rp":
        return 0
    elif type == "sr":
        return []
    else:
        raise NotImplementedError(f"Unsupported task: {type}")