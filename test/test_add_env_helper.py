# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# from src.const import MeasureType
# from src.environment import get_compatible_interface
# from src.environment import CartPoleEnvironment, PendulumEnvironment

# def test_add_env_helper():
    
#     # CartPole (discrete)
#     env = CartPoleEnvironment()
#     print(get_compatible_interface(env, MeasureType.VALUE))
#     print(get_compatible_interface(env, MeasureType.QVALUE))
#     print(get_compatible_interface(env, MeasureType.POLICY))

#     # Pendulum (continuous)
#     env = PendulumEnvironment()
#     print(get_compatible_interface(env, MeasureType.VALUE))
#     print(get_compatible_interface(env, MeasureType.QVALUE))
#     print(get_compatible_interface(env, MeasureType.POLICY))

# def main():
#     test_add_env_helper()

# if __name__ == "__main__":
#     main()