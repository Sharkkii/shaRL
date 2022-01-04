import os
import sys
import numpy as np
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.controller import Controller
from src.environment import CartPoleEnvironment
from src.network import VNet, QNet, PiNet, ValueNetwork, QValueNetwork, PolicyNetwork
# from src.network import get_default_measure_network
# from src.optimizer import get_default_measure_optimizer
# from src.const import MeasureType
from src.optimizer import Optimizer
from src.core import DQN
from src.const import PATH_TO_NETWORK_MODEL

def test_save():

    env = CartPoleEnvironment()

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
        n_epoch = 200,
        n_train_eval = 5,
        n_test_eval = 5,
        env_step = 10
    )

    # save
    path_to_policy = os.path.join(PATH_TO_NETWORK_MODEL, "path_to_policy")
    path_to_value = os.path.join(PATH_TO_NETWORK_MODEL, "path_to_value")
    path_to_qvalue = os.path.join(PATH_TO_NETWORK_MODEL, "path_to_qvalue")
    ctrl.agent.save(
        path_to_policy = path_to_policy,
        path_to_value = path_to_value,
        path_to_qvalue = path_to_qvalue
    )

def test_load(
    do_load
):

    env = CartPoleEnvironment()

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

    # load
    if (do_load):
        path_to_policy = os.path.join(PATH_TO_NETWORK_MODEL, "path_to_policy")
        path_to_value = os.path.join(PATH_TO_NETWORK_MODEL, "path_to_value")
        path_to_qvalue = os.path.join(PATH_TO_NETWORK_MODEL, "path_to_qvalue")
        agent.load(
            path_to_policy = path_to_policy,
            path_to_value = path_to_value,
            path_to_qvalue = path_to_qvalue
        )

    ctrl = Controller(
        environment=env,
        agent=agent,
        config={}
    )

    score = ctrl.evaluate(
        n_train_eval = 5,
        n_test_eval = 5
    )
    print(score)


def test_enable_to_save_model():    

    np.random.seed(0)
    torch.manual_seed(0)
    # test_save()
    test_load(True)
    test_load(False)

    print("OK: test_enable_to_save_model")

def main():
    test_enable_to_save_model()

if __name__ == "__main__":
    main()
