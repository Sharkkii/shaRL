from dataclasses import dataclass, field

@dataclass
class BrainConfig:
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_alpha: float = 1e-4
    tau: float = 0.01
    location: str = "mean" # "max"
    scale: str = "1.0"
    negatizer: str = "zero" # "softplus"
    alpha: str = "1.0"
    beta: str = "1.0"
