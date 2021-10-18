import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# # NOTE: test for RL
from src.environment import Environment, Model, GymEnvironment


def test_add_environment():

    env = GymEnvironment("CartPole-v0")
    observation = env.reset()
    for _ in range(10):
        action = np.random.randint(2)
        observation, _, done, _ = env.step(action)
        print(observation)
        if (done):
            break
    env.update() # warning (not supported)

    env = Model()
    observation = env.reset()
    for _ in range(10):
        action = np.random.randint(2)
        observation, _, done, _ = env.step(action)
        print(observation)
        if (done):
            break
    env.update() # do nothing

    print("OK: test_add_environment")

def main():
    test_add_environment()

if __name__ == "__main__":
    main()
