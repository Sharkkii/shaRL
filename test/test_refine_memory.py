import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.rl_dataset import RLDataset
from src.environment import CartPoleEnvironment

def test_refine_memory():

    env = CartPoleEnvironment()
    buffer = RLDataset(n_capacity = 10)

    for _ in range(10):
        state = env.reset()
        done = False
        trajectory = []
        for _ in range(3):
            if (done): break
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward, next_state))
            state = next_state
        buffer.save(trajectory)
        print(np.array(buffer.dataset["state"]))

    print("OK: test_reinfe_memory")

def main():
    test_refine_memory()

if __name__ == "__main__":
    main()