# import os
# import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# from src.const import MeasureType
# from src.network import get_default_measure_network
# from src.environment import CartPoleEnvironment, PendulumEnvironment

# def test_add_network_helper():
    
#     # CartPole (discrete)
#     env = CartPoleEnvironment()
#     print(get_default_measure_network(env, MeasureType.VALUE).network)
#     print(get_default_measure_network(env, MeasureType.QVALUE).network)
#     print(get_default_measure_network(env, MeasureType.POLICY).network)

#     # Pendulum (continuous)
#     env = PendulumEnvironment()
#     print(get_default_measure_network(env, MeasureType.VALUE).network)
#     print(get_default_measure_network(env, MeasureType.QVALUE).network)
#     print(get_default_measure_network(env, MeasureType.POLICY).network)

# def main():
#     test_add_network_helper()

# if __name__ == "__main__":
#     main()