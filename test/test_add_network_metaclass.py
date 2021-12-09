import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.network import MetaNetwork

def test_add_network_metaclass():
    
    class N(metaclass=MetaNetwork):
        spec = "example"

    print(dir(N))

def main():
    test_add_network_metaclass()

if __name__ == "__main__":
    main()