#### RL Preprocessor ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torchvision.transforms import Compose


class BasePreprocessor(metaclass=ABCMeta):

    def __init__(
        self
    ):
        pass

    def reset(
        self
    ):
        pass

    def setup(
        self
    ):
        pass

    @abstractmethod
    def __call__(
        self
    ):
        raise NotImplementedError


class SequentialPreprocessor(BasePreprocessor):

    def __init__(
        self,
        transforms
    ):
        self.preprocessor = Compose(transforms)
    
    def __call__(
        self,
        x
    ):
        return self.preprocessor(x)


class TensorConverter(BasePreprocessor):

    def __init__(
        self
    ):
        pass

    def __call__(
        self,
        sars
    ):
        state, action, reward, next_state = sars
        state = TensorConverter._to_tensor(state)
        action = TensorConverter._to_tensor(action)
        reward = TensorConverter._to_tensor(reward)
        next_state = TensorConverter._to_tensor(next_state)
        return (state, action, reward, next_state)
    
    def _to_tensor(
        x
    ):
        if (type(x) == np.ndarray):
            return torch.from_numpy(x)
        elif (type(x)) in [int, float, np.int32, np.int64, np.float32, np.float64]:
            return torch.tensor(x)
        else:
            return x

class RewardStabilizer(BasePreprocessor):

    def __init__(
        self,
        alpha = 0.5,
        eps = 1e-4
    ):
        self.target_mu = None
        self.target_sigma = None
        self.alpha = alpha
        self.eps = eps

    def __call__(
        self,
        sars
    ):
        state, action, reward, next_state = sars
        mu = torch.mean(reward)
        sigma = torch.std(reward)
        self.target_mu = mu if (self.target_mu is None) else (self.alpha * mu + (1 - self.alpha) * self.target_mu)
        self.target_sigma = sigma if (self.target_sigma is None) else (self.alpha * sigma + (1 - self.alpha) * self.target_sigma)
        reward = (reward - self.target_mu) / (self.target_sigma + self.eps)
        return (state, action, reward, next_state)