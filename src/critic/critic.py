#### Critic ####

import numpy as np
import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod

import sys
sys.path.append("../src")

from utils.utils_env import unzip_trajectory, zip_trajectory
from utils.utils_value import map_over_trajectory


# base
class Critic(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        value=None,
        qvalue=None,
        optim_value=None,
        optim_qvalue=None,
        smooth_v=0.99,
        smooth_q=0.99,
        gamma=0.99
    ):
        self.value = value
        self.qvalue = qvalue
        self.target_value = value.copy()
        self.target_qvalue = qvalue.copy()
        self.optim_value = optim_value
        self.optim_qvalue = optim_qvalue
        self.smooth_v = smooth_v
        self.smooth_q = smooth_q
        self.gamma = gamma
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def update_value(
        self,
        trajs
    ):
        raise NotImplementedError

    @abstractmethod
    def update_qvalue(
        self,
        trajs
    ):
        raise NotImplementedError

    @abstractmethod
    def update_target_value(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def update_target_qvalue(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def update(
        self
    ):
        self.update_value()
        self.update_qvalue()
        self.update_target_value()
        self.update_target_qvalue()


# default
class DefaultCritic(Critic):

    def __init__(
        self,
        value=None,
        qvalue=None,
        optim_value=None,
        optim_qvalue=None,
        smooth_v=0.99,
        smooth_q=0.99,
        gamma=0.99
    ):
        assert((value is not None) and (qvalue is not None))
        self.value = value
        self.qvalue = qvalue
        self.target_value = value.copy()
        self.target_qvalue = qvalue.copy()
        self.optim_value = optim_value
        self.optim_qvalue = optim_qvalue
        self.smooth_v = smooth_v
        self.smooth_q = smooth_q
        self.gamma = gamma
    
    def reset(self):
        pass

    def setup(self):
        pass

    def update_value(self, trajs):
        pass

    def update_qvalue(self, trajs):
        pass

    def update_target_value(self):
        for theta, target_theta in zip(self.value.parameters(), self.target_value.parameters()):
            target_theta.data = (1 - self.smooth_v) * target_theta.data + self.smooth_v * theta.data

    def update_target_qvalue(self):
        for theta, target_theta in zip(self.qvalue.parameters(), self.target_qvalue.parameters()):
            target_theta.data = (1 - self.smooth_q) * target_theta.data + self.smooth_q * theta.data

    def update(
        self,
        trajs,
        n_times=1
    ):
        for _ in range(n_times):
            J_v = self.update_value(trajs)
            J_q = self.update_qvalue(trajs)
            self.update_target_value()
            self.update_target_qvalue()
        return J_v, J_q


# Q-learning
class QlearningCritic(Critic):

    def __init__(
        self,
        value=None,
        qvalue=None,
        optim_value=None,
        optim_qvalue=None,
        smooth_v=0.01,
        smooth_q=0.01,
        gamma=0.99
    ):
        assert(qvalue is not None)
        # self.value = value
        self.qvalue = qvalue
        # self.target_value = value.copy()
        self.target_qvalue = qvalue.copy()
        # self.optim_value = optim_value
        self.optim_qvalue = optim_qvalue
        # self.smooth_v = smooth_v
        self.smooth_q = smooth_q
        self.gamma = gamma
    
    def reset(self):
        pass

    def setup(self):
        pass

    def update_value(self, trajs):
        J_v = None; return J_v

    @torch.no_grad()
    def update_qvalue(
        self,
        trajs
    ):
        
        # NOTE: trajs: list(list(tuple)) = [[(sars), ...], ...]
        traj = trajs[0]
        n_batch = len(traj)
        state_traj, action_traj, reward_traj, state_next_traj = unzip_trajectory(traj)

        n_state = self.qvalue.qvalue_network.input_shape
        n_action = self.qvalue.qvalue_network.output_shape

        J_q = torch.zeros((n_state, n_action))
        target_q, _ = torch.max(self.qvalue(state_next_traj), dim=1)
        q = map_over_trajectory(self.qvalue, state_traj, action_traj)
        for idx, (state, action, reward, _) in enumerate(traj):
            J_q[state, action] += q[idx] - (reward + self.gamma * target_q[idx])

        self.optim_qvalue.reset()
        # self.optim_qvalue.zero_grad()
        self.optim_qvalue.setup(J_q)
        # J_q.backward()
        self.optim_qvalue.step()
        J_q = torch.mean(J_q)
        return J_q

    def update_target_value(self):
        pass

    def update_target_qvalue(self):
        pass

    def update(
        self,
        trajs,
        n_times=1
    ):
        # TODO: return a seqence of results?
        for _ in range(n_times):
            J_v = self.update_value(trajs)
            J_q = self.update_qvalue(trajs)
            self.update_target_value()
            self.update_target_qvalue()
        return J_v, J_q


# DQN
class DqnCritic(Critic):

    def __init__(
        self,
        value=None,
        qvalue=None,
        optim_value=None,
        optim_qvalue=None,
        smooth_v=0.01,
        smooth_q=0.01,
        gamma=0.99
    ):
        assert(qvalue is not None)
        # self.value = value
        self.qvalue = qvalue
        # self.target_value = value.copy()
        self.target_qvalue = qvalue.copy()
        # self.optim_value = optim_value
        self.optim_qvalue = optim_qvalue
        # self.smooth_v = smooth_v
        self.smooth_q = smooth_q
        self.gamma = gamma
    
    def reset(self):
        pass

    def setup(self):
        pass

    def update_value(self, trajs):
        J_v = None; return J_v

    def update_qvalue(
        self,
        trajs
    ):
        
        # NOTE: trajs: list(list(tuple)) = [[(sars), ...], ...]
        traj = trajs[0]
        n_batch = len(traj)
        state_traj, action_traj, reward_traj, state_next_traj = unzip_trajectory(traj)

        # self.qvalue.qvalue_network.requires_grad = True
        # self.target_qvalue.qvalue_network.requires_grad = False

        with torch.no_grad():
            target_q = self.target_qvalue(state_next_traj)
            target_q, _ = torch.max(self.target_qvalue(state_next_traj), dim=1)
        q = map_over_trajectory(self.qvalue, state_traj, action_traj)
        print(q)
        J_q = F.mse_loss(q, reward_traj + self.gamma * target_q)
        print(J_q)

        self.optim_qvalue.reset()
        # self.optim_qvalue.zero_grad()
        self.optim_qvalue.setup(J_q)
        # J_q.backward()
        self.optim_qvalue.step()
        return J_q

    def update_target_value(self):
        pass
        # for theta, target_theta in zip(self.value.parameters(), self.target_value.parameters()):
        #     target_theta.data = (1 - self.smooth_v) * target_theta.data + self.smooth_v * theta.data

    def update_target_qvalue(self):
        for theta, target_theta in zip(self.qvalue.parameters(), self.target_qvalue.parameters()):
            target_theta.data = (1 - self.smooth_q) * target_theta.data + self.smooth_q * theta.data

    def update(
        self,
        trajs,
        n_times=1
    ):
        # TODO: return a sequence of results?
        for _ in range(n_times):
            J_v = self.update_value(trajs)
            J_q = self.update_qvalue(trajs)
            self.update_target_value()
            self.update_target_qvalue()
        return J_v, J_q


# SAC-discrete
class SacCritic(Critic):

    def __init__(
        self,
        value=None,
        qvalue=None,
        optim_value=None,
        optim_qvalue=None,
        smooth_v=0.01,
        smooth_q=0.01,
        gamma=0.99
    ):
        assert(value is not None)
        assert(qvalue is not None)
        assert(optim_value is not None)
        assert(optim_qvalue is not None)

        self.value = value
        self.qvalue = qvalue
        self.target_value = value.copy()
        self.target_qvalue = qvalue.copy()
        self.optim_value = optim_value
        self.optim_qvalue = optim_qvalue
        self.smooth_v = smooth_v
        self.smooth_q = smooth_q
        self.gamma = gamma
    
    def reset(self):
        pass

    def setup(self):
        pass

    def update_value(
        self,
        trajs,
        policy=None,
        alpha=None
    ):
        assert(policy is not None)
        assert(alpha is not None)
        
        # TODO: organize (-> constant.py?)
        margin = 1e-8
        
        # NOTE: trajs: list(list(tuple)) = [[(sars), ...], ...]
        traj = trajs[0]
        n_batch = len(traj)
        state_traj, action_traj, reward_traj, state_next_traj = unzip_trajectory(traj)

        with torch.no_grad():
            target_q = self.target_qvalue(state_traj)
            pi = torch.clamp(policy(state_traj), margin, 1 - margin)
            log_pi = torch.log(pi)
            expt_q = torch.sum(pi * (target_q - alpha * log_pi), dim=1, keepdim=True)
        
        v = self.value(state_traj)
        J_v = F.mse_loss(v, expt_q)

        # self.optim_value.zero_grad()
        self.optim_value.reset()
        # J_v.backward()
        self.optim_value.setup(J_v)
        self.optim_value.step()
        return J_v

    def update_qvalue(
        self,
        trajs,
        policy=None,
        alpha=None
    ):
        
        # NOTE: trajs: list(list(tuple)) = [[(sars), ...], ...]
        traj = trajs[0]
        n_batch = len(traj)
        state_traj, action_traj, reward_traj, state_next_traj = unzip_trajectory(traj)

        with torch.no_grad():
            target_v = self.target_value(state_next_traj).flatten()
        q = map_over_trajectory(self.qvalue, state_traj, action_traj)
        J_q = F.mse_loss(q, reward_traj + self.gamma * target_v)

        # self.optim_qvalue.zero_grad()
        self.optim_qvalue.reset()
        # J_q.backward()
        self.optim_qvalue.setup(J_q)
        self.optim_qvalue.step()
        return J_q

    def update_target_value(self):
        for theta, target_theta in zip(self.value.parameters(), self.target_value.parameters()):
            target_theta.data = (1 - self.smooth_v) * target_theta.data + self.smooth_v * theta.data

    def update_target_qvalue(self):
        for theta, target_theta in zip(self.qvalue.parameters(), self.target_qvalue.parameters()):
            target_theta.data = (1 - self.smooth_q) * target_theta.data + self.smooth_q * theta.data

    def update(
        self,
        trajs,
        policy=None,
        alpha=None,
        n_times=1
    ):
        assert(policy is not None)
        assert(alpha is not None)
        # TODO: return a sequence of results?
        for _ in range(n_times):
            J_v = self.update_value(trajs, policy, alpha)
            J_q = self.update_qvalue(trajs, policy, alpha)
            self.update_target_value()
            self.update_target_qvalue()
        return J_v, J_q