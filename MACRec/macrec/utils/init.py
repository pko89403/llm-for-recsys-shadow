
import os
import random
import numpy as np
import torch


def init_openai_api(api_config: dict):
    """OpenAI API를 초기화합니다.

    Args:
        api_config (dict): API 구성을 포함하는 딕셔너리입니다.
            - api_base (str): OpenAI API의 기본 URL입니다.
            - api_key (str): OpenAI API의 인증 키입니다.
    """
    # os.environ["OPENAI_API_BASE"] = api_config["api_base"]
    os.environ["OPENAI_API_KEY"] = api_config["api_key"]



def init_all_seeds(seed: int = 0) -> None:
    """
    랜덤 시드를 초기화하는 함수입니다.

    Args:
        seed (int, optional): 랜덤 시드 값입니다. 기본값은 0입니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
