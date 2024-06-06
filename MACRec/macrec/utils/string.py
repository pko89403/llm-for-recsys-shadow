def str2list(s: str) -> list[int]:
    """문자열을 리스트로 변환합니다.

    Args:
        s (str): 변환할 문자열입니다.

    Returns:
        list[int]: 변환된 정수 리스트입니다.
    """

    return [int(i) for i in s.split(',')]