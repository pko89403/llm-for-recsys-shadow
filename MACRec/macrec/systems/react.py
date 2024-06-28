from typing import Any
from loguru import logger

from macrec.systems.base import System
from macrec.agents import Manager
from macrec.utils import parse_answer, parse_action

class ReActSystem(System):
    """
    단일 에이전트(ReAct)를 가진 시스템으로, 여러 동작을 순차적으로 수행할 수 있습니다. 에이전트가 완료되거나 최대 동작 수에 도달하거나 컨텍스트의 제한을 초과하면 시스템이 중지됩니다.
    """
    @staticmethod
    def supported_tasks() -> list[str]:
        return ["rp", "sr", "gen"]
    
    def init(self, *args, **kwargs) -> None:
        """
        ReAct 시스템을 초기화합니다.
        """
        self.manager = Manager(
            thought_config_path=self.config["manager_thought"],
            action_config_path=self.config["manager_action"],
            **self.agent_kwargs)
        self.max_step : int = self.config.get("max_step", 6)
        self.manager_kwargs = dict()
        
    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self.step_n: int = 1
        
    def is_halted(self) -> bool:
        return ((self.step_n > self.max_step) or self.manager.over_limit(input=self.input, scratchpad=self.scratchpad, **self.manager_kwargs)) and not self.finished
    
    def _parse_answer(self, answer: Any = None) -> dict[str, Any]:
        if answer is None:
            answer = self.answer
        return parse_answer(type=self.task, answer=answer, gt_answer=self.gt_answer, json_mode=self.manager.json_mode, **self.kwargs)
    
    def think(self):
        # Think
        logger.debug(f"Step {self.step_n}:")
        self.scratchpad += f"\nThought {self.step_n}:"
        thought = self.manager(
            input=self.input,
            scratchpad=self.scratchpad,
            stage="thought",
            **self.manager_kwargs
        )
        self.scratchpad += " " + thought
        self.log(f"**Thought {self.step_n}**: {thought}", agent=self.manager)
        
    def act(self) -> tuple[str, Any]:
        # Act
        if not self.manager.json_mode:
            # TODO: may be removed after adding more actions
            self.scratchpad += f"\nHint: {self.manager.hint}"
        self.scratchpad += f"\nValid action exampe: {self.manager.valid_action_example}:"
        self.scratchpad += f"\nAction {self.step_n}:"
        action = self.manager(input=self.input, scratchpad=self.scratchpad, stage="action", **self.manager_kwargs)
        self.scratchpad += " " + action
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
        logger.debug(f"Action {self.step_n}| {action}")
        return action_type, argument
        
    def execute(self, action_type: str, argument: Any):
        # Execute
        log_head = ""
        if action_type.lower() == "finish":
            parse_result = self._parse_answer(argument)
        else:
            parse_result = {
                "valid": False,
                "answer": self.answer,
                "message": "Invalid Action type or format."
            }
        if not parse_result["valid"]:
            assert "message" in parse_result, "Invalid parse result."
            observation = f"{parse_result['message']} Valid Action examples are {self.manager.valid_action_example}."
        elif action_type.lower() == "finish":
            observation = self.finish(parse_result["answer"])
            log_head = ':violet[Finish with answer]:\n- '
        else:
            raise ValueError(f"Invalid action type: {action_type}")
    
        self.scratchpad += f"\nObservation: {observation}"
        
        logger.debug(f"Observation: {observation}")
        self.log(f"{log_head}{observation}", agent=self.manager, logging=False)
        
    def step(self):
        self.think()
        action_type, argument = self.act()
        self.execute(action_type, argument)
        self.step_n += 1
        
    def forward(self, reset: bool = True) -> Any:
        if reset:
            self.reset()
        while not self.is_finished() and not self.is_halted():
            self.step()
        return self.answer
            
if __name__ == "__main__":
    import os
    import pandas as pd
    from macrec.utils import init_openai_api, read_json

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    print("Change directory to root path: ", root_dir)
    os.chdir(root_dir)

    init_openai_api(read_json("config/api-config.json"))
    config_path = "config/systems/react/config.json"
    task = "rp"
    logger.info("##### Task:")
    logger.info(f'```\n{task}\n```')

    system = ReActSystem(
        task=task,
        config_path=config_path,
        leak=False,
        web_demo=False,
        dataset="ml-100k",
    )

    data = pd.read_csv("data/ml-100k/test.csv")
    max_his = 5 if task == 'sr' else 10
    data['history'] = data['history'].apply(lambda x: '\n'.join(x.split('\n')[-max_his:]))
    length = len(data)
    index = 0
    reset_data = False
    
    data_sample = data.iloc[index]
    data_prompt = system.prompts[f"data_prompt"]
    logger.info("##### Data Prompt:")
    logger.info(f'```\n{data_prompt}\n```')

    logger.info("##### Target Item Attributes:")
    logger.info(f'```\n{data_sample["target_item_attributes"]}\n```')
    
    gt_answer = data_sample['rating']
    logger.info(f'##### Ground Truth Rating: {gt_answer}')
    system_input = data_prompt.format(
        user_id=data_sample['user_id'],
        user_profile=data_sample['user_profile'],
        history=data_sample['history'],
        target_item_id=data_sample['item_id'],
        target_item_attributes=data_sample['target_item_attributes']
    )
    logger.info('##### Data Prompt:')
    logger.info(f'```\n{system_input}\n```')
    
    system.set_data(input=system_input, context='', gt_answer=gt_answer, data_sample=data_sample)
    system.reset(clear=True)
    
    answer = system()
    logger.info(f'**Answer**: `{answer}`, Ground Truth: `{gt_answer}`')