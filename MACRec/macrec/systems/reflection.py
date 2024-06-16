import json
from typing import Any
from loguru import logger

from macrec.systems.react import ReActSystem
from macrec.agents import Reflector

class ReflectionSystem(ReActSystem):
    """
    매니저와 리플렉터를 가진 시스템으로, 여러 동작을 순차적으로 수행할 수 있습니다. 시스템은 에이전트가 완료되거나 최대 동작 수에 도달하거나 컨텍스트의 제한을 초과할 때 중지됩니다. 시스템은 마지막 시도가 잘못되었다고 판단하면 마지막 시도를 반영합니다.
    """
    def init(self, *args, **kwargs) -> None:
        """
        Initialize the reflection system.
        """
        super().init(*args, **kwargs)
        self.reflector = Reflector(config_path=self.config["reflector"], **self.agent_kwargs)
        self.manager_kwargs["reflections"] = ""
        
    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        super().reset(clear=clear, *args, **kwargs)
        if clear:
            self.reflector.reflections = []
            self.reflector.reflections_str = ""
            
    def forward(self, reset: bool = True) -> Any:
        if self.is_finished() or self.is_halted():
            self.reflector(self.input, self.scratchpad)
            self.reflected = True
            if self.reflector.json_mode:
                reflection_json = json.loads(self.reflector.reflections[-1])
                if "correctness" in reflection_json and reflection_json["correctness"]  == True:
                    # don't forward if the last reflection is correct
                    logger.debug((f"Last reflection is correct, don't forward."))
                    self.log(f":red[**Last reflection is correct, don't forward**]", agent=self.reflector, logging=False)
                    return self.answer
        else:
            self.reflected = False
        self.manager_kwargs["reflections"] = self.reflector.reflections_str
        return super().forward(reset=reset)