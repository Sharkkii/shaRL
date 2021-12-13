#### RL Dataset ####

import numpy as np
import torch
from torch.utils.data import Dataset

class BaseRLDataset:

    def __init__(
        self,
        min_size,
        max_size,
        transform = None
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.transform = transform

    def setup(
        self
    ):
        pass

    def reset(
        self
    ):
        pass

    def load(
        self
    ):
        state = self.dataset["state"]
        action = self.dataset["action"]
        reward = self.dataset["reward"]
        next_state = self.dataset["next_state"]
        history = BaseRLDataset.zip(state, action, reward, next_state)
        return history
    
    def save(
        self,
        history
    ):
        state, action, reward, next_state = BaseRLDataset.unzip(history)
        self.dataset["state"].extend(state)
        self.dataset["action"].extend(action)
        self.dataset["reward"].extend(reward)
        self.dataset["next_state"].extend(next_state)
        self._discard()

    def _discard(
        self
    ):
        for key in self.dataset.keys():
            self.dataset[key] = self.dataset[key][- self.max_size:]

    # {(s, a, r, s_next)} -> ({s}, {a}, {r}, {s_next})
    def unzip(
        history
    ):
        state_traj, action_traj, reward_traj, state_next_traj = zip(*history)
        return state_traj, action_traj, reward_traj, state_next_traj
    
    def unzip_or_nop(
        history
    ):
        if (type(history) == list):
            return BaseRLDataset.unzip(history)
        elif (type(history) == tuple):
            return history

    # ({s}, {a}, {r}, {s_next}) -> {(s, a, r, s_next)}
    def zip(
        state_traj,
        action_traj,
        reward_traj,
        state_next_traj
    ):
        history = list(zip(state_traj, action_traj, reward_traj, state_next_traj))
        return history

    def zip_or_nop(
        history
    ):
        if (type(history) == tuple):
            return BaseRLDataset.zip(*history)
        elif (type(history) == list):
            return history

class RLDataset(Dataset, BaseRLDataset):

    def __init__(
        self,
        min_size,
        max_size,
        transform = None
    ):
        BaseRLDataset.__init__(
            self,
            min_size = min_size,
            max_size = max_size,
            transform = transform
        )
        self.dataset = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": []
        }

    @property
    def is_available(
        self
    ):
        return (self.min_size <= len(self.dataset["state"]))

    def __len__(
        self
    ):
        l = len(self.dataset["state"])
        if (l < self.min_size):
            l = self.min_size
        elif (l > self.max_size):
            l = self.max_size
        return l

    def __getitem__(
        self,
        index
    ):
        index = index % len(self)
        state = self.dataset["state"][index]
        action = self.dataset["action"][index]
        reward = self.dataset["reward"][index]
        next_state = self.dataset["next_state"][index]
        if (self.transform is not None):
            state, action, reward, next_state = self.transform((state, action, reward, next_state))
        return (state, action, reward, next_state)
