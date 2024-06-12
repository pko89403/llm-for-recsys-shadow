from typing import Callable

def run_once(func: Callable) -> Callable:
    """한 번만 실행되는 함수를 위한 데코레이터입니다.
    
    Args:
        `func` (`Callable`): 데코레이션할 함수입니다.
    
    Returns:
        `Callable`: 데코레이션된 함수입니다.
    """
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)
    wrapper.has_run = False
    return wrapper