#### Actor ####

import numpy as np
import torch
from abc import ABCMeta, abstractmethod

import sys
sys.path.append("../src")

from policy import ValueBasedPolicy, EpsilonGreedyDecoder
from utils.utils_env import unzip_trajectory, zip_trajectory
from utils.utils_value import map_over_trajectory


# base
class Actor(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        policy=None,
        optim_policy=None
    ):
        self.policy = policy
        self.behavior_policy = None
        self.target_policy = None
        self.optim_policy = optim_policy

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    def interact(
        self,
        env,
        is_discrete_action_space=True, # FIXME: combine into env.
        is_deterministic_policy=True,
        n_times=1,
        n_limit=1000,
    ):

        trajs = []
        for _ in range(n_times):

            t = 0
            state = env.reset()
            done = False
            traj = []
            for _ in range(n_limit):

                if (done): break
                if (is_discrete_action_space):
                    x = self.policy(state).detach().numpy()
                    if (is_deterministic_policy):
                        action = np.argmax(x)
                    else:
                        action = np.random.choice(env.action_space.n, p=x)

                else:
                    # FIXME: support other continuous action policies than gaussian
                    x = self.policy(state).detach().numpy()
                    mu, sigma = x[[0]], x[[1]]
                    if (is_deterministic):
                        action = mu
                    else:
                        eps = np.random.randn(1)
                        action = mu + eps * sigma

                state_next, reward, done, info = env.step(action)
                traj.append((state, action, reward, state_next))
                state = state_next

            trajs.append(traj)

        return trajs


    @abstractmethod
    def update_behavior_policy(
        self,
        trajs=None,
        value=None,
        qvalue=None,
        target_value=None,
        target_qvalue=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def update_target_policy(
        self,
        trajs=None,
        value=None,
        qvalue=None,
        target_value=None,
        target_qvalue=None,
    ):
        raise NotImplementedError

    def update(
        self,
        trajs=None,
        value=None,
        qvalue=None,
        n_times=1
    ):
        for _ in range(n_times):
            J_pi = self.update_behavior_policy()
            J_pi = self.update_target_policy()
        return J_pi


# default
class DefaultActor(Actor):

    def __init__(
        self,
        policy,
        optim_policy=None
    ):
        self.policy = policy
        self.behavior_policy = None
        self.target_policy = None
        self.optim_policy = optim_policy

    def reset(self):
        self.policy.reset()

    def setup(self):
        self.policy.setup()

    def update_behavior_policy(self,trajs=None,n_times=1):
        pass

    def update_target_policy(self,trajs=None,n_times=1):
        pass


# Q-learning
class QlearningActor(Actor):
    def __init__(
        self,
        policy,
        optim_policy=None,
        eps=0.0
    ):
        super().__init__(policy, optim_policy)
        self.eps = eps
    
    def reset(self):
        self.policy.reset()

    def setup(
        self,
        eps=None
    ):
        if (eps is not None):
            assert(0.0 <= eps <= 1.0)
            self.eps = eps
        self.policy.decoder = EpsilonGreedyDecoder(eps=self.eps)

    def update_behavior_policy(
        self,
        trajs=None,
        value=None,
        qvalue=None,
        target_value=None,
        target_qvalue=None,
    ):
        J_pi = None; return J_pi

    def update_target_policy(
        self,
        trajs=None,
        value=None,
        qvalue=None,
        target_value=None,
        target_qvalue=None,
    ):
        J_pi = None; return J_pi


# DQN
DqnActor = QlearningActor


# SAC-discrete
class SacActor(Actor):
    def __init__(
        self,
        policy,
        optim_policy=None
    ):
        assert(policy is not None)
        assert(optim_policy is not None)
        super(SacActor, self).__init__(
            policy=policy,
            optim_policy=optim_policy
        )
    
    def reset(self):
        self.policy.reset()

    def setup(
        self,
    ):
        pass
        # if (eps is not None):
        #     assert(0.0 <= eps <= 1.0)
        #     self.eps = eps
        # self.policy.decoder = EpsilonGreedyDecoder(eps=self.eps)

    def update_behavior_policy(
        self,
        trajs=None,
        value=None,
        qvalue=None,
        target_value=None,
        target_qvalue=None,
        alpha=None
    ):
        J_pi = None; return J_pi

    def update_target_policy(
        self,
        trajs=None,
        value=None,
        qvalue=None,
        target_value=None,
        target_qvalue=None,
        alpha=None
    ):
        assert(value is not None)
        assert(qvalue is not None)
        assert(alpha is not None)

        # TODO: organize (-> constant.py?)
        margin = 1e-8
        
        traj = trajs[0]
        n_batch = len(traj)
        state_traj, action_traj, reward_traj, state_next_traj = unzip_trajectory(traj)

        with torch.no_grad():
            v = value(state_traj)
            q = qvalue(state_traj)

        pi = torch.clamp(self.policy(state_traj), margin, 1 - margin)
        log_pi = torch.log(pi)
        J_pi = torch.mean(torch.sum(pi * (alpha * log_pi - q), dim=1))

        self.optim_policy.reset()
        # self.optim_policy.zero_grad()
        self.optim_policy.setup(J_pi)
        # J_pi.backward()
        self.optim_policy.step()
        return J_pi

    def update(
        self,
        trajs=None,
        value=None,
        qvalue=None,
        alpha=None,
        n_times=1
    ):
        assert(value is not None)
        assert(qvalue is not None)
        assert(alpha is not None)

        for _ in range(n_times):
            J_pi = self.update_behavior_policy(
                trajs,
                value=value,
                qvalue=qvalue,
                alpha=alpha
            )
            J_pi = self.update_target_policy(
                trajs,
                value=value,
                qvalue=qvalue,
                alpha=alpha
            )
        return J_pi