import pandas as pd
import streamlit as st
from abc import ABC, abstractmethod
from typing import Any, Optional
from loguru import logger
from langchain.prompts import PromptTemplate

from macrec.agents import Agent
from macrec.utils import is_correct, init_answer, read_json, read_prompts, get_avatar, get_color

class System(ABC):
    """
    시스템의 기본 클래스입니다. 
    `forward` 함수를 사용하여 시스템 출력을 얻습니다. 
    `set_data`를 사용하여 입력, 문맥 및 정답을 설정합니다. 
    `is_finished`를 사용하여 시스템이 완료되었는지 확인합니다. 
    `is_correct`를 사용하여 시스템 출력이 올바른지 확인합니다. 
    `finish`를 사용하여 시스템을 완료하고 시스템 출력을 설정합니다.
    """
    
    @staticmethod
    @abstractmethod
    def supported_tasks() -> list[str]:
        """
        지원하는 작업 목록을 반환합니다.

        Returns:
            list[str]: 지원하는 작업 목록
        """
        raise NotImplementedError("System.supported_tasks() not implemented")
    
    @property
    def task_type(self) -> str:
        """작업 유형을 반환합니다. 하위 클래스에서 상속하여 더 많은 작업 유형을 지원할 수 있습니다.
        
        Raises:
            `NotImplementedError`: 지원되지 않는 작업 유형입니다.
        Returns:
            `str`: 작업의 유형입니다.
        하위 클래스 예시:
        .. code-block:: python
            class MySystem(System):
            @property
            def task_type(self) -> str:
                if self.task == 'my_task':
                return '내 작업 설명'
                else:
                return super().task_type
        """

        if self.task == "qa":
            return "question_answering"
        elif self.task == "rp":
            return "rating_prediction"
        elif self.task == "sr":
            return "ranking"
        elif self.task == "chat":
            return "conversion"
        elif self.task == "gen":
            return "explanation generation"
        else:
            raise NotImplementedError

    def __init__(self, task: str, config_path: str, leak: bool = False, web_demo: bool = False, dataset: Optional[str] = None, *args, **kwargs) -> None:
        """클래스의 초기화 메서드입니다.

        Args:
            task (str): 작업의 이름입니다.
            config_path (str): 설정 파일의 경로입니다.
            leak (bool, optional): 데이터 누수 여부입니다. 기본값은 False입니다.
            web_demo (bool, optional): 웹 데모 여부입니다. 기본값은 False입니다.
            dataset (Optional[str], optional): 데이터셋의 이름입니다. 기본값은 None입니다.
            *args: 추가적인 위치 인수입니다.
            **kwargs: 추가적인 키워드 인수입니다.
        """
        self.task = task
        assert self.task in self.supported_tasks()
        self.config = read_json(config_path)
        if "supported_tasks" in self.config:
            assert isinstance(self.config["supported_tasks"], list) and self.task in self.config["supported_tasks"], f"Task {self.task} is not supported by the system."
        self.agent_kwargs = {
            "system": self,
        }
        if dataset is not None:
            for key, value in self.config.items():
                if isinstance(value, str):
                    self.config[key] = value.format(dataset=dataset, task=self.task)
            self.agent_kwargs["dataset"] = dataset
        self.prompts = read_prompts(self.config["agent_prompt"])
        self.prompts.update(read_prompts(self.config["data_prompt"].format(task=self.task)))
        if "task_agent_prompt" in self.config:
            self.prompts.update(read_prompts(self.config["task_agent_prompt"].format(task=self.task)))
        self.agent_kwargs["prompts"] = self.prompts
        self.leak = leak
        self.web_demo = web_demo
        self.agent_kwargs["web_demo"] = web_demo
        self.kwargs = kwargs
        self.init(*args, **kwargs)
        self.reset(clear=True)

    def log(self, message: str, agent: Optional[Agent] = None, logging: bool = True) -> None:
        """로그를 기록합니다.

        Args:
            message (str): 로그 메시지입니다.
            agent (Optional[Agent], optional): 에이전트입니다. 기본값은 None입니다.
            logging (bool, optional): 로깅 여부입니다. 기본값은 True입니다.
        """
        if logging: 
            logger.debug(message)
        if self.web_demo:
            if agent is None:
                role = "Assistant"
            else:
                role = agent.__class__.__name__
            final_message = f"{get_avatar(role)}:{get_color(role)}[**{role}**]: {message}"
            if "manager" not in role.lower() and "assistant" not in role.lower():
                messsages = final_message.split("\n")
                messages = [f"- {messages[0]}"] + [f"  {message}" for message in messsages[1:]]
                final_message = "\n".join(messages)
            self.web_log.append(final_message)
            st.markdown(f"{final_message}")

    @abstractmethod
    def init(self, *args, **kwargs) -> None:
        """클래스의 초기화 메서드입니다.

        Args:
            *args: 추가적인 위치 인수입니다.
            **kwargs: 추가적인 키워드 인수입니다.
        """
        raise NotImplementedError("System.init() not implemented")
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.clear_web_log()
        return self.forward(*args, **kwargs)
    
    def set_data(self, input: str, context: str, gt_answer: Any, data_sample: Optional[pd.Series] = None) -> None:
        self.input: str = input
        self.context: str = context
        self.gt_answer = gt_answer
        self.data_sample = data_sample

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("System.forward() not implemented")
    
    def is_finished(self) -> bool:
        return self.finished
    
    def is_correct(self) -> bool:
        return is_correct(task=self.task, answer=self.answer, gt_answer=self.gt_answer)
    
    def finish(self, answer: Any) -> str:
        self.answer = answer
        if not self.leak:
            observation = f"The answer you give (may be INCORRECT): {self.answer}"
        elif self.is_correct():
            observation = "Answer is CORRECT"
        else:
            observation = "Answer is INCORRECT"
        self.finished = True
        return observation
    
    def clear_web_log(self) -> None:
        self.web_log = []

    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        self.scratchpad: str = ""
        self.finished: bool = False
        self.answer = init_answer(type=self.task)
        if self.web_demo and clear:
            self.clear_web_log()