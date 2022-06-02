# import torch
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from src.network import VNet, QNet, PiNet, ValueNetwork, QValueNetwork, PolicyNetwork, DefaultNetwork
# from src.environment import CartPoleEnvironment
# from src.core import DQN
# from src.controller import Controller
# from src.optimizer import Optimizer


# def test_check_network():

#     v_net = VNet(input_shape=4)
#     v_opt = Optimizer(torch.optim.Adam, lr=1e-2)
#     v_net = ValueNetwork(value_network=v_net)

#     q_net = QNet(input_shape=4, output_shape=2)
#     q_opt = Optimizer(torch.optim.Adam, lr=1e-4)
#     q_net = QValueNetwork(qvalue_network=q_net)

#     pi_net = PiNet(input_shape=4, output_shape=2)
#     pi_opt = Optimizer(torch.optim.Adam, lr=1e-4)
#     pi_net = PolicyNetwork(policy_network=pi_net)

#     net = DefaultNetwork(
#         d_in = 4,
#         d_out = 2
#     )
#     print("OK: test_check_network")

# def test_fix_controller():

#     env = CartPoleEnvironment()

#     v_net = VNet(input_shape=4)
#     v_opt = Optimizer(torch.optim.Adam, lr=1e-2)
#     v_net = ValueNetwork(value_network=v_net)

#     q_net = QNet(input_shape=4, output_shape=2)
#     q_opt = Optimizer(torch.optim.Adam, lr=1e-4)
#     q_net = QValueNetwork(qvalue_network=q_net)

#     pi_net = PiNet(input_shape=4, output_shape=2)
#     pi_opt = Optimizer(torch.optim.Adam, lr=1e-4)
#     pi_net = PolicyNetwork(policy_network=pi_net)

#     agent = DQN(
#         gamma = 0.90,
#         tau = 0.01,
#         eps = 0.4,
#         eps_decay = 1.0
#     )
#     agent.setup(
#         env = env,
#         policy_network = pi_net,
#         value_network = v_net,
#         qvalue_network = q_net,
#         policy_optimizer = pi_opt,
#         value_optimizer = v_opt,
#         qvalue_optimizer = q_opt
#     )

#     ctrl = Controller(
#         environment=env,
#         agent=agent,
#         config={}
#     )
#     train_score, test_score = ctrl.fit(
#         n_epoch = 100,
#         n_train_eval = 5,
#         n_test_eval = 5,
#         env_step = 10,
#         dataset_size = 1000,
#         batch_size = 100,
#         shuffle = True,
#         return_score = True
#     )
#     print(train_score)
#     print(test_score)

# def test_refactor_all():
#     test_check_network()
#     test_fix_controller()
#     print("OK: test_refactor_all")

# def main():
#     test_refactor_all()

# if __name__ == "__main__":
#     main()
