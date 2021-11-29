# Value

## Interface

```Python
class Value()

  def __init__(
    self
  ) -> None

  def reset(
    self
  ) -> None

  def setup(
    self,
    value_network: ValueNetwork,
    value_optimizer: Optimizer
  ) -> None

  def __call__(
    self,
    state: State
  ) -> torch.tensor
  where State = torch.tensor

class QValue()

  def __init__(
    self
  ) -> None

  def reset(
    self
  ) -> None

  def setup(
    self,
    qvalue_network: QValueNetwork,
    qvalue_optimizer: Optimizer
  ) -> None

  def __call__(
    self,
    state: State,
    action: Action
  ) -> torch.tensor
  where State = torch.tensor, Action = torch.tensor
```