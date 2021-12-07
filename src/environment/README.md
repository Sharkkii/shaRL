# Environment

## Interface

```Python
class Environment()

  def __init__(
    self
  ) -> None

  def reset(
    self
  ) -> Observation
  where Observation = numpy.ndarray

  def step(
    self,
    action: Action
  ) -> tuple[Observation, Reward, bool, T]
  where Observation = numpy.ndarray, Reward = numpy.ndarray, T = Any

  def score(
    self,
    trajectory: list[tuple[State, Action, Reward, State]]
  ) -> T
  where State = torch.tensor, Action = torch.tensor, Reward = torch.tensor, T = Any
```