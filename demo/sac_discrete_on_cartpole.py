import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch.nn as nn
import torch.optim as optim

from src.network import DiscreteQValueNetwork
from src.network import DiscretePolicyNetwork
from src.optimizer import Optimizer
from src.environment import CartPoleEnvironment
from src.core import SoftActorCriticDiscrete
from src.controller import Controller

def main():

    # define NN
    class Network(nn.Module):

        def __init__(self, input_shape, output_shape):
            super().__init__()
            self.input_shape = input_shape
            self.output_shape = output_shape
            self.l1 = nn.Linear(self.input_shape, 20)
            self.l2 = nn.Linear(20, 20)
            self.l3 = nn.Linear(20, self.output_shape)
            self.reset()

        def reset(self):
            nn.init.normal_(self.l1.weight, mean=0., std=1.0)
            nn.init.normal_(self.l2.weight, mean=0., std=1.0)
            nn.init.normal_(self.l3.weight, mean=0., std=1.0)
            
        def forward(self, x):
            x = nn.ReLU()(self.l1(x))
            x = nn.ReLU()(self.l2(x))
            x = self.l3(x)
            return x

    # create function approximators (NN)
    policy_network = DiscretePolicyNetwork(
        policy_network = Network(4, 2)
    )
    policy_optimizer = Optimizer(optim.Adam, lr=1e-2)
    qvalue_network = DiscreteQValueNetwork(
        qvalue_network = Network(4, 2)
    )
    qvalue_optimizer = Optimizer(optim.Adam, lr=1e-2)

    # create Environment & Agent
    env = CartPoleEnvironment()
    agent = SoftActorCriticDiscrete(
        gamma = 0.90,
        alpha = 1.0,
        alpha_decay = 1.0,
        tau = 0.01
    )
    agent.setup(
        env = env,
        policy_network = policy_network,
        policy_optimizer = policy_optimizer,
        qvalue_network = qvalue_network,
        qvalue_optimizer = qvalue_optimizer
    )

    # Run RL
    controller = Controller(
        environment = env,
        agent = agent
    )
    train_score, test_score = controller.fit(
        n_epoch = 1000,
        env_step = 10,
        dataset_size = 10000,
        batch_size = 100,
        n_train_eval = 5,
        n_test_eval = 5,
        return_score = True
    )
    print(train_score)
    print(test_score)

if __name__ == "__main__":
    main()
