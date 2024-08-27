import numpy as np
import torch

def set_deterministic(seed):
    """
    Sets seeds for numpy and torch.
    """
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)