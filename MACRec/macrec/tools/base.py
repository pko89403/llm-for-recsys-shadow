from abc import ABC, abstractmethod

from macrec.utils import read_json


class Tool(ABC):
    """
    도구에 대한 기본 클래스입니다.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        self.config = read_json(config_path)
        
    @abstractmethod
    def reset(self) -> None:
        """
        도구를 초기 상태로 재설정합니다.
        """
        raise NotImplementedError("reset method not implemented")
    
class RetrieverTool(Tool):
    """검색 도구 클래스입니다.

    Args:
        Tool (_type_): 도구 클래스입니다.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs):
            
    @abstractmethod
    def search(self, query: str) -> str:
        raise NotImplementedError("search method not implemented")
    
    @abstractmethod
    def lookup(self, title: str, term: str) -> str:
        raise NotImplementedError("lookup method not implemented")