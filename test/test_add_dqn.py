import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.controller import Controller
from src.environment import CartPoleEnvironment
from src.network import VNet, QNet, PiNet, ValueNetwork, QValueNetwork, PolicyNetwork
from src.optimizer import Optimizer
from src.core import DQN


def test_add_dqn():

    v_net = VNet(input_shape=4)
    v_opt = Optimizer(torch.optim.Adam, lr=1e-2)
    v_net = ValueNetwork(value_network=v_net)

    q_net = QNet(input_shape=4, output_shape=2)
    q_opt = Optimizer(torch.optim.Adam, lr=1e-4)
    q_net = QValueNetwork(qvalue_network=q_net)

    pi_net = PiNet(input_shape=4, output_shape=2)
    pi_opt = Optimizer(torch.optim.Adam, lr=1e-4)
    pi_net = PolicyNetwork(policy_network=pi_net)

    env = CartPoleEnvironment()

    agent = DQN(
        gamma = 0.99,
        tau = 0.01,
        eps = 0.25,
        eps_decay = 0.999
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
        environment=env,
        agent=agent,
        config={}
    )
    ctrl.fit(
        n_epoch = 1,
        n_sample = 100,
        n_sample_start = 100,
        n_train_eval = 5,
        n_test_eval = 0,
        env_step = 1,
        gradient_step = 10
    )

    score = ctrl.evaluate(
        n_train_eval = 10,
        n_test_eval = 1
    )
    print(score)

    print("OK: test_add_dqn")

def main():
    test_add_dqn()

if __name__ == "__main__":
    main()
