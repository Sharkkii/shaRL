#### Helper ####

import torch.optim as optim
from ..optimizer import Optimizer

def get_default_measure_optimizer():
    kwargs = {}
    default_measure_optimizer = Optimizer(
        optimizer = optim.Adam,
        **kwargs
    )
    return default_measure_optimizer
