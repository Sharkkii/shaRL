import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn

from src.network import MetaNetwork

def test_add_network_metaclass():
    
    class N(nn.Module, metaclass=MetaNetwork):
        spec = "example_network"

    n = N(d_in = 5, d_out = 1)
    x = torch.randn(size=(10, 5))
    y = n(x)
    print(x, y)

def main():
    test_add_network_metaclass()

if __name__ == "__main__":
    main()