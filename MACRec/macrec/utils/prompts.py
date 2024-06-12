import os
import json
from langchain.prompts import PromptTemplate

def read_prompts(config_file: str) -> dict[str, PromptTemplate | str]:
    """지정된 구성 파일에서 프롬프트를 읽어옵니다.

    Args:
        config_file (str): 구성 파일의 경로

    Returns:
        dict[str, PromptTemplate | str]: 프롬프트 이름과 템플릿 또는 문자열로 구성된 딕셔너리
    """
    
    assert os.path.exists(config_file), f"config_file {config_file} does not exist"
    with open(config_file, "r") as f:
        config = json.load(f)
    ret = {}
    for prompt_name, prompt_config in config.items():
        assert "content" in prompt_config
        if "type" not in prompt_config:
            template = PromptTemplate.from_template(template=prompt_config["content"])
            if template.input_variables == []:
                template = template.template
        elif prompt_config["type"] == "raw":
            template = prompt_config["content"]
        elif prompt_config["type"] == "template":
            template = PromptTemplate.from_template(template=prompt_config["content"]) 
        ret[prompt_name] = template
    return ret