# import torch
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from src.controller import Controller
# from src.environment import CartPoleEnvironment
# from src.network import VNet, QNet, PiNet, ValueNetwork, DiscreteQValueNetwork, PolicyNetwork
# from src.optimizer import Optimizer
# from src.core import SoftActorCriticDiscrete


# def test_add_sac_discrete():

#     env = CartPoleEnvironment()

#     v_net = VNet(input_shape=4)
#     v_opt = Optimizer(torch.optim.Adam, lr=1e-3)
#     v_net = ValueNetwork(value_network=v_net)

#     q_net = QNet(input_shape=4, output_shape=2)
#     q_opt = Optimizer(torch.optim.Adam, lr=1e-3)
#     q_net = DiscreteQValueNetwork(qvalue_network=q_net)

#     pi_net = PiNet(input_shape=4, output_shape=2)
#     pi_opt = Optimizer(torch.optim.Adam, lr=1e-3)
#     pi_net = PolicyNetwork(policy_network=pi_net)

#     agent = SoftActorCriticDiscrete(
#         gamma = 0.90,
#         alpha = 1.0,
#         alpha_decay = 1.0,
#         tau = 0.01
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
#         environment = env,
#         agent = agent,
#         config = {}
#     )

#     ctrl.fit(
#         n_epoch = 1000,
#         # n_sample = 100,
#         # n_sample_start = 100,
#         n_train_eval = 5,
#         n_test_eval = 5,
#         env_step = 10,
#         # gradient_step = -1
#     )

#     train_score, test_score = ctrl.evaluate(
#         n_train_eval = 5,
#         n_test_eval = 5
#     )
#     print(train_score["duration"], test_score["duration"])

#     print("OK: test_add_sac_discrete")

# def main():
#     test_add_sac_discrete()

# if __name__ == "__main__":
#     main()
