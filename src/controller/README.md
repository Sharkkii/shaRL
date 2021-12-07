# Controller

## Interface

```Python
class Controller()

  def __init__(
    self,
    environment: Environment,
    agent: Agent
  ) -> None

  def reset(
    self
  ) -> None

  def setup(
    self
  ) -> None

  def fit(
    self,
    n_epoch: int,
    n_sample: int,
    n_sample_start: int,
    n_train_eval: int,
    n_test_eval: int,
    env_step: int,
    gradient_step: int
  ) -> None

  def evaluate(
    self,
    n_train_eval: int,
    n_test_eval: int
  ) -> (dict[string, list[T]])
  where T = TypeVar

```