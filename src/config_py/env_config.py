from dataclasses import dataclass

@dataclass
class EnvConfig:
    task: str = "MountainCar-discrete"
    # task: "Pendulum-discrete"
    # task: "Pendulum-continuous"
    # task: "CartPole-discrete"