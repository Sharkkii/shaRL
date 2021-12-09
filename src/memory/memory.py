#### Memory ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch


class BaseMemory(metaclass=ABCMeta):

    def __init__(
        self
    ):
        raise NotImplementedError
    
    def reset(
        self
    ):
        raise NotImplementedError

    def setup(
        self
    ):
        raise NotImplementedError
    
    def save(
        self,
        x
    ):
        raise NotImplementedError
    
    def load(
        self
    ):
        raise NotImplementedError
    
    def replay(
        self,
        n_sample = 0
    ):
        raise NotImplementedError


class Memory(BaseMemory):

    def __init__(
        self,
        capacity = 10000
    ):
        assert(capacity > 0)
        self._capacity = capacity
        self.cell = []
        self._count = 0

    @property
    def capacity(self):
        return self._capacity
    
    @capacity.setter
    def capacity(self, capacity):
        self._capacity = capacity
    
    @property
    def count(self):
        return len(self.cell)
    
    @count.setter
    def count(self, count):
        pass
    
    def reset(
        self
    ):
        self.cell = []
        self._count = 0

    def setup(
        self
    ):
        pass
    
    def save(
        self,
        x
    ):
        self.cell = self.cell + x
        self.cell = self.cell[-self._capacity:]
    
    def load(
        self
    ):
        return self.cell
    
    def replay(
        self,
        n_sample = 0
    ):
        return np.random.choice(self.cell, n_sample)
    
    def get(
        self,
        index = None
    ):
        return self.cell if (index is None) else [self.cell[i] for i in index]

class RLMemory(BaseMemory):

    def __init__(
        self,
        capacity = 10000
    ):
        self.state_memory = Memory(
            capacity = capacity
        )
        self.action_memory = Memory(
            capacity = capacity
        )
        self.reward_memory = Memory(
            capacity = capacity
        )
        self.next_state_memory = Memory(
            capacity = capacity
        )
        self.reward_processor = RewardProcessor(
            alpha = 0.5
        )
    
    @property
    def capacity(self):
        return self.state_memory.capacity
    
    @capacity.setter
    def capacity(self, capacity):
        self.state_memory.capacity = capacity
        self.action_memory.capacity = capacity
        self.reward_memory.capacity = capacity
        self.next_state_memory.capacity = capacity

    @property
    def count(self):
        return self.state_memory.count
    
    @count.setter
    def count(self, count):
        pass
    
    def reset(
        self
    ):
        self.state_memory.reset()
        self.action_memory.reset()
        self.reward_memory.reset()
        self.next_state_memory.reset()
    
    def setup(
        self
    ):
        self.state_memory.setup()
        self.action_memory.setup()
        self.reward_memory.setup()
        self.next_state_memory.setup()

    def save(
        self,
        history
    ):
        state_trajectory, action_trajectory, reward_trajectory, next_state_trajectory = RLMemory.unzip(history)
        self.state_memory.save(state_trajectory.tolist())
        self.action_memory.save(action_trajectory.tolist())
        self.reward_memory.save(reward_trajectory.tolist())
        self.next_state_memory.save(next_state_trajectory.tolist())
    
    def load(
        self,
        do_zip = False
    ):
        state_trajectory = torch.tensor(self.state_memory.load())
        action_trajectory = torch.tensor(self.action_memory.load())
        reward_trajectory = torch.tensor(self.reward_memory.load())
        next_state_trajectory = torch.tensor(self.next_state_memory.load())
        if (do_zip):
            history = RLMemory.zip(
                state_trajectory,
                action_trajectory,
                reward_trajectory,
                next_state_trajectory
            )
        else:
            history = (
                state_trajectory,
                action_trajectory,
                reward_trajectory,
                next_state_trajectory
            )
        return history
    
    def replay(
        self,
        n_sample = 0,
        do_zip = False,
        do_normalize = True
    ):
        assert(self.count >= n_sample)
        index = list(np.random.randint(0, self.count, n_sample))
        state_trajectory = torch.tensor(self.state_memory.get(index))
        action_trajectory = torch.tensor(self.action_memory.get(index))
        reward_trajectory = torch.tensor(self.reward_memory.get(index))
        next_state_trajectory = torch.tensor(self.next_state_memory.get(index))

        if (do_normalize):
            reward_trajectory = self.reward_processor(
                reward_trajectory
            )

        if (do_zip):
            history = RLMemory.zip(
                state_trajectory,
                action_trajectory,
                reward_trajectory,
                next_state_trajectory
            )
        else:
            history = (
                state_trajectory,
                action_trajectory,
                reward_trajectory,
                next_state_trajectory
            )
        return history

    # {(s, a, r, s_next)} -> ({s}, {a}, {r}, {s_next})
    def unzip(
        trajectory
    ):

        state_traj, action_traj, reward_traj, state_next_traj = zip(*trajectory)
        state_traj = torch.from_numpy(np.array(state_traj).astype(np.float32))
        action_traj = torch.from_numpy(np.array(action_traj).astype(np.float32))
        reward_traj = torch.from_numpy(np.array(reward_traj).astype(np.float32))
        state_next_traj = torch.from_numpy(np.array(state_next_traj).astype(np.float32))

        return state_traj, action_traj, reward_traj, state_next_traj

    # ({s}, {a}, {r}, {s_next}) -> {(s, a, r, s_next)}
    def zip(
        state_traj,
        action_traj,
        reward_traj,
        state_next_traj
    ):
        
        state_traj = state_traj.detach().numpy()
        action_traj = action_traj.detach().numpy()
        reward_traj = reward_traj.detach().numpy()
        state_next_traj = state_next_traj.detach().numpy()

        return list(zip(state_traj, action_traj, reward_traj, state_next_traj))

class RewardProcessor:

    def __init__(
        self,
        alpha = 0.5
    ):
        self.target_mu = None
        self.target_sigma = None
        self.alpha = alpha
    
    def __call__(
        self,
        reward,
        eps = 1e-4
    ):
        mu = torch.mean(reward)
        sigma = torch.std(reward)
        self.target_mu = mu if (self.target_mu is None) else (self.alpha * mu + (1 - self.alpha) * self.target_mu)
        self.target_sigma = sigma if (self.target_sigma is None) else (self.alpha * sigma + (1 - self.alpha) * self.target_sigma)

        reward = (reward - self.target_mu) / (self.target_sigma + eps)
        return reward
        