import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.controller import Controller
from src.environment import CartPoleEnvironment
from src.network import VNet, QNet, PiNet, ValueNetwork, QValueNetwork, PolicyNetwork
# from src.network import get_default_measure_network
# from src.optimizer import get_default_measure_optimizer
# from src.const import MeasureType
from src.optimizer import Optimizer
from src.core import DQN


def test_add_dqn():

    env = CartPoleEnvironment()
    
    # v_net = get_default_measure_network(env, MeasureType.VALUE)
    # v_opt = get_default_measure_optimizer()

    # q_net = get_default_measure_network(env, MeasureType.QVALUE)
    # q_opt = get_default_measure_optimizer()

    # pi_net = get_default_measure_network(env, MeasureType.POLICY)
    # pi_opt = get_default_measure_optimizer()

    v_net = VNet(input_shape=4)
    v_opt = Optimizer(torch.optim.Adam, lr=1e-2)
    v_net = ValueNetwork(value_network=v_net)

    q_net = QNet(input_shape=4, output_shape=2)
    q_opt = Optimizer(torch.optim.Adam, lr=1e-2)
    q_net = QValueNetwork(qvalue_network=q_net)

    pi_net = PiNet(input_shape=4, output_shape=2)
    pi_opt = Optimizer(torch.optim.Adam, lr=1e-2)
    pi_net = PolicyNetwork(policy_network=pi_net)

    agent = DQN(
        gamma = 0.90,
        tau = 0.01,
        eps = 0.4,
        eps_decay = 1.0
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
        n_epoch = 1000,
        # n_sample = 100,
        # n_sample_start = 100,
        n_train_eval = 5,
        n_test_eval = 5,
        env_step = 10,
        # gradient_step = -1
    )

    score = ctrl.evaluate(
        n_train_eval = 5,
        n_test_eval = 5
    )
    print(score)

    print("OK: test_add_dqn")

def main():
    test_add_dqn()

if __name__ == "__main__":
    main()
