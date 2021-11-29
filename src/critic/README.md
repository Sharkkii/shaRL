# Critic

## Interface

```Python
class Critic()

    def __init__(
      self
    ) -> None

    def reset(
      self
    ) -> None

    def setup(
      self,
      value_network: ValueNetwork,
      qvalue_network: QValueNetwork,
      value_optimizer: Optimizer,
      qvalue_optimizer: Optimizer
    ) -> None

    def update_value(
      self,
      actor: Actor,
      trajectory: list[tuple[State, Action, Reward, State]]
    ) -> None
    where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

    def update_qvalue(
      self,
      actor: Actor,
      trajectory: list[tuple[State, Action, Reward, State]]
    ) -> None
    where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

    def update_target_value(
      self,
      actor: Actor,
      trajectory: list[tuple[State, Action, Reward, State]]
    ) -> None
    where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

    def update_target_qvalue(
      self,
      actor: Actor,
      trajectory: list[tuple[State, Action, Reward, State]]
    ) -> None
    where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

    def update(
      self,
      actor: Actor,
      trajectory: list[tuple[State, Action, Reward, State]],
      n_times: int
    ) -> None
    where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor
```