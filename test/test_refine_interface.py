# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import gym

# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# # # NOTE: test for RL
# from src.controller import Controller
# from src.environment import Environment, Model
# from src.agent import *
# from src.actor import *
# from src.critic import *
# from src.policy import Policy
# from src.value import Value, QValue
# from src.network import VNet, QNet, PiNet, ValueNetwork, QValueNetwork, PolicyNetwork
# from src.memory import Memory
# from src.optimizer import Optimizer


# def test_refine_interface():

#     v_net = VNet(input_shape=2)
#     v_net = ValueNetwork(value_network=v_net)
#     v_opt = Optimizer()
#     value = Value(v_net, v_opt)
#     q_net = QNet(input_shape=2, output_shape=2)
#     q_net = QValueNetwork(qvalue_network=q_net)
#     q_opt = Optimizer()
#     qvalue = QValue(q_net, q_opt)
#     pi_net = PiNet(input_shape=2, output_shape=2)
#     pi_net = PolicyNetwork(policy_network=pi_net)
#     pi_opt = Optimizer()
#     policy = Policy(pi_net, pi_opt)

#     actor = Actor(
#         policy=policy
#     )
#     critic = Critic(
#         value=value,
#         qvalue=qvalue
#     )

#     memory = Memory()
#     env = Environment() # PseudoMountainCar()
#     model = Model()
#     agent = Agent(
#         actor=actor,
#         critic=critic,
#         model=model,
#         memory=memory
#     )
#     # agent.setup(env)

#     ctrl = Controller(
#         environment=env,
#         agent=agent,
#         config={}
#     )
#     ctrl.fit(
#         n_eval=1,
#         n_eval_interval=1
#     )

#     print("OK: test_refine_interface")

# def main():
#     test_refine_interface()

# if __name__ == "__main__":
#     main()
