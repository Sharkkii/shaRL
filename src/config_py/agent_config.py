from dataclasses import dataclass

@dataclass
class AgentConfig:
    n_epoch: int = 10000
    n_batch: int = 100
    n_memory: int = 10000
    env_step: int = 1
    gradient_step: int = 10
    eval_interval: int = 50
    n_eval: int = 5
    gamma: float = 0.99