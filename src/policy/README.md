# Policy

## Interface

```Python
class Policy()

  def __init__(
    self
  ) -> None

  def reset(
    self
  ) -> None

  def setup(
    self,
    policy_network: PolicyNetwork,
    policy_optimizer: Optimizer
  ) -> None

  def __call__(
    self,
    state: State
  )
  where State = torch.tensor
```