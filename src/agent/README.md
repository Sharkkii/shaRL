# Agent

## Interface

```Python
class Agent()
  
  def __init__(
    self,
    model: Model,
    memory: Memory,
    gamma: float
  ) -> None

  def reset(
    self
  ) -> None

  def setup(
    self,
    env: Environment,
    policy_network: PolicyNetwork,
    value_network: ValueNetwork,
    qvalue_network: QValueNetwork,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    qvalue_optimizer: Optimizer
  ) -> None

  def update_model(
    self,
    trajectory: list[tuple[State, Action, Reward, State]],
    n_times: int
  ) -> None
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

  def update_actor(
    self,
    trajectory: list[tuple[State, Action, Reward, State]],
    n_times: int
  ) -> None
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

  def update_critic(
    self,
    trajectory: list[tuple[State, Action, Reward, State]],
    n_times: int
  ) -> None
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

  def interact_with(
    self,
    env: Environment
  ) -> list[tuple[State, Action, Reward, State]]
  where State = numpy.ndarray, Action = numpy.ndarray, Reward = numpy.ndarray

  def replay_history(
    self,
    n_sample: int
  ) -> list[tuple[State, Action, Reward, State]]
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

  def load_history(
    self,
  ) -> list[tuple[State, Action, Reward, State]]
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

  def save_history(
    self,
    history: list[tuple[State, Action, Reward, State]]
  ) -> None
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor
```