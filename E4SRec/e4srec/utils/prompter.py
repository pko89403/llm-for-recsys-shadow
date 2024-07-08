import os
import json
from typing import Union, List


class Prompter(object):
    __slots__ = ("template", "_verbose") # template, _verbose 변수를 가짐
    
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # 여기에서 기본값을 강제로 설정하여, 생성자가 ''로 호출되어도 오류가 발생하지 않도록 합니다.
            template_name = "alpaca"
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
    
    def generate_prompt(
        self,
        task_type: str,
    ) -> List[str]:
        # 지시문과 선택적 입력을 포함한 전체 프롬프트를 반환합니다.
        # 레이블(=응답, =출력)이 제공되면 이를 추가합니다.
        if task_type == "general":
            instruction = "Given the user ID and purchase history, predict the most suitable item for the user."
        elif task_type == "sequential":
            instruction = "Given the user's purchase history, predict next possible item to be purchased."
        else:
            instruction = ""
        ins = self.template["prompt_input"].format(
            instruction=instruction
        )
        res = self.template["response_split"]
        if self._verbose:
            print(ins + res)
        return [ins, res]