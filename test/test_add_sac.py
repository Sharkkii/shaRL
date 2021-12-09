import numpy as np
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.controller import Controller
from src.environment import GymEnvironment, PendulumEnvironment
from src.network import VNet, QNet, PiNet, ValueNetwork, ContinuousQValueNetwork, GaussianPolicyNetwork
from src.optimizer import Optimizer
from src.core import SoftActorCritic


def test_add_sac():

    np.random.seed(0)
    torch.manual_seed(0)

    v_net = VNet(input_shape=3)
    v_opt = Optimizer(torch.optim.Adam, lr=1e-3)
    v_net = ValueNetwork(value_network=v_net)

    q_net = QNet(input_shape=4, output_shape=1)
    q_opt = Optimizer(torch.optim.Adam, lr=1e-3)
    q_net = ContinuousQValueNetwork(qvalue_network=q_net)

    pi_net_mu = PiNet(input_shape=3, output_shape=1)
    pi_net_sigma = PiNet(input_shape=3, output_shape=1)
    pi_opt = Optimizer(torch.optim.Adam, lr=1e-4)
    pi_net = GaussianPolicyNetwork(
        network_mu = pi_net_mu,
        network_sigma = pi_net_sigma
    )

    env = PendulumEnvironment()

    agent = SoftActorCritic(
        gamma = 0.99,
        alpha = 1.0,
        alpha_decay = 1.0,
        tau = 0.01
    )
    agent.setup(
        env = env,
        policy_network = pi_net,
        value_network = v_net,
        qvalue_network = q_net,
        policy_optimizer = pi_opt,
        value_optimizer = v_opt,
        qvalue_optimizer = q_opt
    )

    ctrl = Controller(
        environment = env,
        agent = agent,
        config = {}
    )

    ctrl.fit(
        n_epoch = 100,
        n_sample = 1000,
        n_sample_start = 1000,
        n_train_eval = 5,
        n_test_eval = 1,
        env_step = 1,
        gradient_step = 10
    )

    train_score, test_score = ctrl.evaluate(
        n_train_eval = 5,
        n_test_eval = 1
    )
    print(train_score["total_reward"], test_score["total_reward"])

    print("OK: test_add_sac")

def main():
    test_add_sac()

if __name__ == "__main__":
    main()
