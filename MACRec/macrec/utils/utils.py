from typing import TypeVar

T = TypeVar("T")

def get_rm(d: dict, key: str, value: T) -> T:
    """사전에서 키를 가져오고 제거합니다.
    
    Args:
        `d` (`dict`): 사전.
        `key` (`str`): 가져오고 제거할 키.
        `value` (`T`): 키가 없을 경우 반환할 값.
    Returns:
        `T`: 키가 있는 경우 해당 값, 그렇지 않으면 인수로 전달된 값.
    """
    ret = d.get(key, value)
    if key in d:
        del d[key]
    return ret

def task2name(task: str) -> str:
    """작업 약어를 해당 작업의 전체 이름으로 변환합니다.
    
    Args:
        `task` (`str`): 작업 약어.
    Returns:
        `str`: 작업의 전체 이름.
    """
    if task == "rp":
        return "Rating Prediction"
    elif task == "sr":
        return "Sequential Recommendation"
    elif task == "gen":
        return "Explanation Generation"
    elif task == "chat":
        return "Conversational Recommendation"
    else:
        raise ValueError(f"Task {task} is not supported.")

def system2dir(system: str) -> str:
    """시스템 이름을 해당 디렉토리 이름으로 변환합니다.
    
    Args:
        `system` (`str`): 시스템 이름.
    Returns:
        `str`: 시스템의 디렉토리 이름.
    """
    assert "system" in system.lower(), "The system name should contain 'system'!"
    return system.lower().replace("system", "")