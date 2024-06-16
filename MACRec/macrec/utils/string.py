def format_step(step: str) -> str:
    """단계 프롬프트를 형식화합니다. 선행 및 후행 공백과 개행을 제거하고 개행을 공백으로 대체합니다.
    
    Args:
        `step` (`str`): 문자열 형식의 단계 프롬프트입니다.
    Returns:
        `str`: 형식화된 단계 프롬프트입니다.
    """
    return step.strip("\n").strip().replace("\n", "")

def format_last_attempt(input: str, scratchpad: str, header: str) -> str:
    """마지막 시도의 반영 프롬프트를 형식화합니다. `scratchpad`의 앞뒤 공백과 개행을 제거하고 개행을 공백으로 대체합니다. `header`를 프롬프트의 시작에 추가합니다.
    
    Args:
        `input` (`str`): 마지막 시도의 입력입니다.
        `scratchpad` (`str`): 마지막 시도의 스크래치패드입니다.
        `header` (`str`): 마지막 시도 반영 헤더입니다.
    Returns:
        `str`: 형식화된 마지막 시도 프롬프트입니다.
    """
    return header + f"Input:\n{input}\n" + scratchpad.strip("\n").strip() + "\n(END PREVIOUS TRAIL)\n"

def format_reflections(reflections: list[str], header: str) -> str:
    """
    주어진 형식에 맞게 reflection 프롬프트를 포맷합니다. 각 reflection의 앞뒤 공백과 개행 문자를 제거하고, 개행 문자를 공백으로 대체합니다. 프롬프트의 시작에 `header`를 추가합니다.

    Args:
        `reflections` (`list[str]`): 이전 reflection의 리스트입니다.
        `header` (`str`): reflection 헤더입니다.

    Returns:
        `str`: 포맷된 reflection 프롬프트입니다. 만약 `reflections`가 비어있는 경우 빈 문자열을 반환합니다.
    """
    if reflections == []:
        return ''
    else:
        return header + "Reflections:\n- " + "\n- ".join([r.strip() for r in reflections])

def format_history(history: list[dict]) -> str:
    """히스토리를 포맷합니다. `history`의 각 턴 사이에 새 줄을 추가합니다.
    
    인자:
        `history` (`list[dict]`): 히스토리의 턴들로 구성된 리스트입니다. 각 턴은 `command`와 `observation` 키를 가진 딕셔너리입니다.
    반환값:
        `str`: 포맷된 히스토리 프롬프트입니다. `history`가 비어있는 경우 빈 문자열을 반환합니다.
    """
    if history == []:
        return ""
    else:
        return "\n" + "\n".join([f"Command: {turn['command']}\nObservation: {turn['observation']}\n" for turn in history]) + "\n"

def format_chat_history(history: list[tuple[str, str]]) -> str:
    """채팅 기록을 형식화합니다. `history`의 각 턴 사이에 새 줄을 추가합니다.
    
    Args:
        `history` (`list[tuple[str, str]]`): 채팅 기록의 턴 목록입니다. 각 턴은 채팅 기록과 역할이라는 두 요소로 구성된 튜플입니다.
    Returns:
        `str`: 형식화된 채팅 기록입니다. `history`가 비어있는 경우 `'No chat history.\\n'`를 반환합니다.
    """
    if history == []:
        return "No chat history.\n"
    else:
        return "\n" + "\n".join([f"{role.capitalize()}: {chat}" for chat, role in history]) + "\n"


def str2list(s: str) -> list[int]:
    """문자열을 정수 리스트로 변환합니다.
    
    인수:
        `s` (`str`): 쉼표로 구분된 정수들로 이루어진 문자열입니다. 예를 들어, `'1,2,3'`입니다.
    반환값:
        `list[int]`: 정수들로 이루어진 리스트입니다. 예를 들어, `[1, 2, 3]`입니다.
    """
    return [int(i) for i in s.split(',')]

def get_avatar(agent_type: str) -> str:
    """에이전트의 아바타를 가져옵니다.
    
    Args:
        `agent_type` (`str`): 에이전트의 타입입니다.
    Returns:
        `str`: 에이전트의 아바타입니다.
    """
    if 'manager' in agent_type.lower():
        return '👩‍💼'
    elif 'reflector' in agent_type.lower():
        return '👩‍🔬'
    elif 'searcher' in agent_type.lower():
        return '🔍'
    elif 'interpreter' in agent_type.lower():
        return '👩‍🏫'
    elif 'analyst' in agent_type.lower():
        return '👩‍💻'
    else:
        return '🤖'
