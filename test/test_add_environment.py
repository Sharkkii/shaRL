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
from src.environment import Environment, Model, GymEnvironment, CartPoleEnvironment, PendulumEnvironment


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

    env = CartPoleEnvironment()
    observation = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        print(action)
        observation, reward, done, _ = env.step(action)
        print(observation, reward)
        if (done):
            break

    env = PendulumEnvironment()
    observation = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        print(action)
        observation, reward, done, _ = env.step(action)
        print(observation, reward)
        if (done):
            break

    print("OK: test_add_environment")

def main():
    test_add_environment()

if __name__ == "__main__":
    main()
