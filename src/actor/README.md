# Actor

## Interface

```Python
class Actor()

  def __init__(
    self,
    policy: Policy
  ) -> None

  def reset(
    self
  ) -> None

  def setup(
    self,
    policy_network: PolicyNetwork,
    policy_optimizer: Optimizer
  ) -> None

  def choose_action(
    self,
    state: State,
    action_space: Union[gym.spaces.Discrete, gym.spaces.Box]
  ) -> Action
  where State = numpy.ndarray, Action = numpy.ndarray

  def update_policy(
    self,
    trajectory: list[tuple[State, Action, Reward, State]]
  ) -> None
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

  def update_target_policy(
    self,
    trajectory: list[tuple[State, Action, Reward, State]]
  ) -> None
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor

  def update(
    self,
    trajectory: list[tuple[State, Action, Reward, State]],
    n_times: int
  ) -> None
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor


```