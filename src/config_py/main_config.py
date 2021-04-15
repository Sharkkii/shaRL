from dataclasses import dataclass

from .env_config import EnvConfig
from .agent_config import AgentConfig
from .brain_config import BrainConfig
from .value_config import ValueConfig
from .policy_config import PolicyConfig
from .net_config import NetConfig

@dataclass
class MainConfig:
    env: EnvConfig = EnvConfig()
    agent: AgentConfig = AgentConfig()
    brain: BrainConfig = BrainConfig()
    value: ValueConfig = ValueConfig()
    policy: PolicyConfig = PolicyConfig()
    net: NetConfig = NetConfig()

def main():
    import hydra
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    cs.store(name="main_config", node=MainConfig)

