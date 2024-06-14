import json 
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, Optional, TYPE_CHECKING
from langchain.prompts import PromptTemplate

from macrec.llms import BaseLLM, AnyOpenAILLM, OpenSourceLLM
from macrec.tools import TOOL_MAP, Tool
from macrec.utils import run_once, format_history, read_prompts

if TYPE_CHECKING:
    from macrec.systems import System

class Agent(ABC):
    def __init__(self, prompts: dict = dict(), prompt_config: Optional[str] = None, web_demo: bool = False, system: Optional["System"] = None, dataset: Optional[str] = None, *args, **kwargs) -> None:
        """Agent를 초기화합니다.
        
        매개변수:
            `prompts` (`dict`, 선택적): Agent에 대한 프롬프트의 사전입니다. `prompt_config`가 `None`이 아닌 경우 프롬프트 구성 파일에서 읽어옵니다. 기본값은 `dict()`입니다.
            `prompt_config` (`Optional[str]`): 프롬프트 구성 파일의 경로입니다. 기본값은 `None`입니다.
            `web_demo` (`bool`, 선택적): Agent가 웹 데모에서 사용되는지 여부입니다. 기본값은 `False`입니다.
            `system` (`Optional[System]`): Agent가 속한 시스템입니다. 기본값은 `None`입니다.
            `dataset` (`Optional[str]`): Agent가 사용되는 데이터셋입니다. 기본값은 `None`입니다.
        """
        self.json_mode: bool
        self.system = system
        if prompt_config is not None:
            prompts = read_prompts(prompt_config)
        self.prompts = prompts
        if self.system is not None:
            for prompt_name, prompt_template in self.prompts.items():
                if isinstance(prompt_template, PromptTemplate) and "task_type" in prompt_template.input_variables:
                    self.prompts[prompt_name] = prompt_template.partial(task_type=self.system.task_type)
        self.web_demo = web_demo
        self.dataset = dataset
        if self.web_demo:
            assert self.system is not None, "System not found."

    def observation(self, message: str, log_head: str = "") -> None:
        """메시지를 기록합니다.
        
        Args:
            `message` (`str`): 기록할 메시지입니다.
            `log_head` (`str`): 로그 헤드입니다. 기본값은 `''`입니다.
        """

        if self.web_demo:
            self.system.log(log_head + message, agent=self)
        else:
            logger.debug(f"Observation: {message}")
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Agent의 전방향 패스입니다.
        
        예외:
            `NotImplementedError`: 서브클래스에서 구현되어야 합니다.
        반환값:
            `Any`: Agent의 출력값.
        """
        raise NotImplementedError("Agent.forward() not implemented")

    def get_LLM(self, config_path: Optional[str] = None, config: Optional[dict] = None) -> BaseLLM:
        """Agent의 기본 대형 언어 모델을 가져옵니다.
        
        매개변수:
            `config_path` (`Optional[str]`): LLM의 구성 파일 경로입니다. `config`가 `None`이 아닌 경우 이 인자는 무시됩니다. 기본값은 `None`입니다.
            `config` (`Optional[dict]`): LLM의 구성입니다. 기본값은 `None`입니다.
        반환값:
            `BaseLLM`: LLM 객체입니다.
        """
        if config is None:
            assert config_path is not None
            with open(config_path, "r") as f:
                config = json.load(f)
        model_type = config["model_type"]
        del config["model_type"]
        if model_type != "api":
            return OpenSourceLLM(**config)
        else:
            return AnyOpenAILLM(**config)


class ToolAgent(Agent):
    """
    도구를 필요로하는 에이전트들의 기본 클래스입니다. `forward` 함수를 사용하여 에이전트의 출력을 얻습니다. `required_tools`를 사용하여 에이전트에 필요한 도구를 지정합니다.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tools: dict[str, Tool] = {}
        self._history = []
        self.max_turns: int = 6
        
    @run_once
    def validate_tools(self) -> None:
        """에이전트가 필요로 하는 도구들을 검증합니다.
        
        Raises:
            `AssertionError`: 필요한 도구가 발견되지 않은 경우.
        """
        required_tools = self.required_tools()
        for tool, tool_type in required_tools.items():
            assert tool in self.tools, f"도구 {tool}을(를) 찾을 수 없습니다."
            assert isinstance(self.tools[tool], tool_type), f"도구 {tool}은(는) {tool_type}의 인스턴스여야 합니다."
    
    @staticmethod
    @abstractmethod
    def required_tools() -> dict[str, type]:
        """에이전트에 필요한 도구들을 반환합니다.
        
        Raises:
            `NotImplementedError`: 서브클래스에서 구현되어야 합니다.
        Returns:
            `dict[str, type]`: 필요한 도구들의 이름과 타입입니다.
        """
        raise NotImplementedError("Agent.required_tools() not implemented")

    def get_tools(self, tool_config: dict[str, dict]):
        assert isinstance(tool_config, dict), "Tool config must be a directory."
        for tool_name, tool in tool_config.items():
            assert isinstance(tool, dict), "Config of each tool must be a dictionary."
            assert "type" in tool, "Tool type not found."
            assert "config_path" in tool, "Tool config path not found."
            tool_type = tool["type"]
            if tool_type not in TOOL_MAP:
                raise NotImplementedError(f"Docstore {tool_type} not implemented.")
            config_path = tool["config_path"]
            if self.dataset is not None:
                config_path = config_path.format(dataset=self.dataset)
            self.tools[tool_name] = TOOL_MAP[tool_type](config_path=config_path)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.validate_tools()
        self.reset()
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def invoke(self, argument: Any, json_mode: bool) -> str:
        """에이전트를 인자와 함께 호출합니다.
        
        매개변수:
            `argument` (`Any`): 에이전트에 대한 인자입니다.
            `json_mode` (`bool`): 인자가 JSON 모드인지 여부입니다.
        예외:
            `NotImplementedError`: 서브클래스에서 구현되어야 합니다.
        반환값:
            `str`: 호출 과정의 관찰 결과입니다.
        """
        raise NotImplementedError("Agent.invoke() not implemented")        

    def reset(self) -> None:
        self._history = []
        self.finished = False
        self.results = None
        for tool in self.tools.values():
            tool.reset()
    
    @property
    def history(self) -> str:
        return format_history(self._history)

    def finish(self, results: Any) -> str:
        self.results = results
        self.finished = True
        return str(self.results)
    
    def is_finished(self) -> bool:
        return self.finished or len(self._history) >= self.max_turns
    
