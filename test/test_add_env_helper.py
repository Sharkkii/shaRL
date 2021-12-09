import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.environment.helper import get_compatible_interface
from src.environment import CartPoleEnvironment, PendulumEnvironment

def test_add_env_helper():
    env = CartPoleEnvironment()
    print(get_compatible_interface(env))
    env = PendulumEnvironment()
    print(get_compatible_interface(env))

def main():
    test_add_env_helper()

if __name__ == "__main__":
    main()