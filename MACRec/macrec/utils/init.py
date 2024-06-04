
import os
import random
import numpy as np
import torch

def init_all_seeds(seed: int = 0) -> None:
    """_summary_

    Args:
        seed (int, optional): 랜덤 시드. Defaults to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
