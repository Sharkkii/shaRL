import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.controller import Controller
from src.environment import PendulumEnvironment
from src.network import VNet, QNet, PiNet, ValueNetwork, ContinuousQValueNetwork, PolicyNetwork
from src.optimizer import Optimizer
from src.core import DeepDeterministicPolicyGradient


def test_add_ddpg():

    v_net = VNet(input_shape=3)
    v_opt = Optimizer(torch.optim.Adam, lr=1e-4)
    v_net = ValueNetwork(value_network=v_net)

    q_net = QNet(input_shape=4, output_shape=1)
    q_opt = Optimizer(torch.optim.Adam, lr=1e-4)
    q_net = ContinuousQValueNetwork(qvalue_network=q_net)

    pi_net = PiNet(input_shape=3, output_shape=1)
    pi_opt = Optimizer(torch.optim.Adam, lr=1e-4)
    pi_net = PolicyNetwork(policy_network=pi_net)

    env = PendulumEnvironment()

    agent = DeepDeterministicPolicyGradient(
        gamma = 0.99,
        tau = 0.01,
        k = 5.0
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
        n_sample = 200,
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

    print("OK: test_add_ddpg")

def main():
    test_add_ddpg()

if __name__ == "__main__":
    main()
