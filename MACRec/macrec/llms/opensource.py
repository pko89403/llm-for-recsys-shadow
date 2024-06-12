import json
from multiprocessing import Pipe
from jsonformer import Jsonformer
from loguru import logger
from typing import Any
from transformers import pipeline
from transformers.pipelines import Pipeline

from macrec.llms.basellm import BaseLLM

class MyJsonFormer:
    """
    A JSON Former that uses a pipeline to generate JSON.
    
    """
    def __init__(self, json_schema: dict, pipeline: Pipeline, max_new_tokens: int = 300, temperature: float = 0.9, debug: bool = False):
        """Initialize the JSON Former.

        Args:
            json_schema (dict): The JSON schema to use.
            pipeline (Pipeline): The pipeline of the LLM. Must be a `pipeline("text-generation")`.
            max_new_tokens (int, optional): . Defaults to 300.
            temperature (float, optional): _description_. Defaults to 0.9.
            debug (bool, optional): . Defaults to False.
        """
        self.json_schema = json_schema
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the JSON Former.
        
        Args:
            prompt (str): The prompt to use.
            **kwargs (Any): Additional arguments.

        Returns:
            str: The generated JSON.
        """

        model = Jsonformer(
            model=self.pipeline.model,
            tokenizer=self.pipeline.tokenizer,
            json_schema=self.json_schema,
            prompt=prompt,
            max_number_tokens=self.max_new_tokens,
            max_string_token_length=self.max_new_tokens,
            debug=self.debug,
            temperature=self.temperature
        )
        text = model()
        return json.dumps(text, ensure_ascii=False)
        

class OpenSourceLLM(BaseLLM):
    def __init__(self, model_path: str = "lmsys/vicuna-7b-v1.5-16k", device: int = 0, json_mode: bool = False, prefix: str = "react", max_new_tokens: int = 300, do_sample: bool = True, temperature: float = 0.9, top_p: float = 1.0, *args, **kwargs):
        """클래스의 인스턴스를 초기화합니다.

        Args:
            model_path (str, optional): 모델 경로입니다. Defaults to "lmsys/vicuna-7b-v1.5-16k".
            device (int, optional): 디바이스 번호입니다. Defaults to 0.
            json_mode (bool, optional): JSON 모드 여부입니다. Defaults to False.
            prefix (str, optional): 접두사입니다. Defaults to "react".
            max_new_tokens (int, optional): 최대 새 토큰 수입니다. Defaults to 300.
            do_sample (bool, optional): 샘플링 여부입니다. Defaults to True.
            temperature (float, optional): 온도 값입니다. Defaults to 0.9.
            top_p (float, optional): Top-p 값입니다. Defaults to 1.0.
        """
        self.json_mode = json_mode
        if device == "auto":
            self.pipe = pipeline("text-generation", model=model_path, device_map="auto")
        else:
            self.pipe = pipeline("text-generation", model=model_path, device=device)
        self.pipe.model.generation_config.do_sample = do_sample
        self.pipe.model.generation_config.top_p = top_p
        self.pipe.model.generation_config.temperature = temperature
        self.pipe.model.generation_config.max_new_tokens = max_new_tokens

        if self.json_mode:
            logger.info("Enabling JSON mode")
            json_schema = kwargs.get(f"{prefix}_json_schema", None)
            assert json_schema is not None, "json_schema must be provided if json_mode is True"
            self.pipe = MyJsonFormer(json_schema=json_schema, pipeline=self.pipe, max_new_tokens=max_new_tokens, temperature=temperature, debug=kwargs.get("debug", False))
        self.model_name = model_path
        self.max_tokens = max_new_tokens
        self.max_context_length: int = 16384 if "16k" in model_path else 32768 if "32k" in model_path else 4096

    def __call__(self, prompt: str, *args, **kwargs) -> str: # type: ignore
        """사용자가 입력한 프롬프트에 대한 응답을 생성하는 함수입니다.

        Args:
            prompt (str): 사용자에게 제공된 프롬프트입니다.

        Returns:
            str: 생성된 응답입니다.
        """
        if self.json_mode:
            return self.pipe.invoke(prompt)
        else:
            return self.pipe.invoke(prompt, return_full_text=False)[0]["generated_text"]

