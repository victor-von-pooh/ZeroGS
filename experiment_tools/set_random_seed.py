import random

import numpy as np
import torch


def fix_seed(seed: int) -> None:
    """
    乱数シードを固定する関数

    Parameters
    ----------
    seed: int
        乱数シードの値
    Returns
    ----------
    None
    """
    # Random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
