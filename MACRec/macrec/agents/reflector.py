import tiktoken
from enum import Enum
from loguru import logger
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate

from macrec.agents.base import Agent
from macrec.llms import AnyOpenAILLM
from macrec.utils import format_step, format_reflections, format_last_attempt, read_json, get_rm

class ReflectionStrategy(Enum):
    """
    `Reflector` 에이전트의 반사 전략입니다. 
    `NONE`은 반사를 하지 않는 것을 의미합니다.
    `REFLEXION`은 `Reflector` 에이전트의 기본 전략으로, LLM에게 입력과 스크래치패드에 대해 반사하도록 요청합니다.
    `LAST_ATTEMPT`는 단순히 마지막 시도의 입력과 스크래치패드를 저장합니다.
    `LAST_ATTEMPT_AND_REFLEXION`은 두 전략을 결합한 것입니다.
    """    
    NONE = "base"
    LAST_ATTEMPT = "last_trial"
    REFLEXION = "reflection"
    LAST_ATTEMPT_AND_REFLEXION = "last_trial_and_reflection"
    
class Reflector(Agent):
    """
    반사 에이전트입니다. 반사 에이전트는 기본적으로 LLM에게 입력과 스크래치패드에 대해 반사하도록 요청합니다. 다른 반사 전략도 지원됩니다. 자세한 내용은 `ReflectionStrategy`를 참조하세요.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        """반사 에이전트를 초기화합니다. 반사 에이전트는 기본적으로 LLM에게 입력과 스크래치패드에 대해 반사하도록 요청합니다.
        
        Args:
            `config_path` (`str`): 반사 에이전트의 설정 파일 경로입니다.
        """
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        keep_reflection = get_rm(config, "keep_reflections", True)
        reflection_strategy = get_rm(config, "reflection_strategy", ReflectionStrategy.REFLEXION.value)
        self.llm = self.get_LLM(config=config)
        if isinstance(self.llm, AnyOpenAILLM):
            self.enc = tiktoken.encoding_for_model(self.llm.model_name)
        else:
            self.enc = AutoTokenizer.from_pretrained(self.llm.model_name)
        self.json_mode = self.llm.json_mode
        self.keep_reflection = keep_reflection
        for strategy in ReflectionStrategy:
            if strategy.value == reflection_strategy:
                self.reflection_strategy = strategy
                break
        assert self.reflection_strategy is not None, f"Unknown reflection strategy: {reflection_strategy}"
        self.reflections: list[str] = []
        self.reflections_str: str = ""
        
    @property
    def reflector_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts["reflect_prompt_json"]
        else:
            return self.prompts["reflect_prompt"]
        
    @property
    def reflect_examples(self) -> str:
        prompt_name = "reflect_examples_json" if self.json_mode else "reflect_examples"
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            return ""
        
    def _build_reflector_prompt(self, input: str, scratchpad: str) -> str:
        return self.reflector_prompt.format(
            examples=self.reflect_examples,
            input=input,
            scratchpad=scratchpad
        )

    def _prompt_reflection(self, input: str, scratchpad: str) -> str:
        reflection_prompt = self._build_reflector_prompt(input, scratchpad)
        reflection_response = self.llm(reflection_prompt)
        if self.keep_reflection:
            self.reflection_input = reflection_prompt
            self.reflection_output = reflection_response
            logger.trace(f"Reflection input length: {len(self.enc.encode(self.reflection_input))}")
            logger.trace(f"Reflection input: {self.reflection_input}")
            logger.trace(f"Reflection output length: {len(self.enc.encode(self.reflection_output))}")
            if self.json_mode:
                self.system.log(f"[:violet[Reflection]]:\n- `{self.reflection_output}`", agent=self, logging=False)
            else:
                self.system.log(f"[:violet[Reflection]]:\n- {self.reflection_output}", agent=self, logging=False)
            logger.debug(f"Reflection: {self.reflection_output}")
        return format_step(reflection_response)
    
    def forward(self, input: str, scratchpad: str, *args, **kwargs) -> str:
        logger.trace("Running Reflection strategy...")
        if self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT:
            self.reflections = [scratchpad]
            self.reflections_str = format_last_attempt(input, scratchpad, self.prompts["last_trial_header"])
        elif self.reflection_strategy == ReflectionStrategy.REFLEXION:
            self.reflections.append(self._prompt_reflection(input=input, scratchpad=scratchpad))
            self.reflections_str = format_reflections(self.reflections, header=self.prompts["reflection_header"])
        elif self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(input, scratchpad, self.prompts["last_trial_header"])
            self.reflections = self._prompt_reflection(input=input, scratchpad=scratchpad)
            self.reflections_str += format_reflections(self.reflections, header=self.prompts["reflection_last_trial_header"])
        elif self.reflection_strategy == ReflectionStrategy.NONE:
            self.reflections = []
            self.reflections_str = ""
        else:
            raise ValueError(f"Unknown reflection strategy: {self.reflection_strategy}")
        logger.trace(self.reflections_str)
        return self.reflections_str