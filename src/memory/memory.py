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
    
    # def push_back(
    #     self,
    #     x
    # ):
    #     raise NotImplementedError
    
    # def pop_back(
    #     self
    # ):
    #     raise NotImplementedError
    
    # def push_front(
    #     self,
    #     x
    # ):
    #     raise NotImplementedError

    # def pop_front(
    #     self
    # ):
    #     raise NotImplementedError
    
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
        n_sample = 1
    ):
        raise NotImplementedError


class Memory(BaseMemory):

    def __init__(
        self,
        capacity = 100
    ):
        assert(capacity > 0)
        self.capacity = capacity
        self.cell = []
        self.count = 0
    
    def reset(
        self
    ):
        self.cell = []
        self.count = 0

    def setup(
        self
    ):
        pass
    
    def save(
        self,
        x
    ):
        pass
    
    def load(
        self
    ):
        return []
    
    def replay(
        n_sample = 1
    ):
        return []
    
        

        