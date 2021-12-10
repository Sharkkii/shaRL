import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.value import get_default_value
from src.value import get_default_qvalue
from src.policy import get_default_policy
from src.environment import CartPoleEnvironment, PendulumEnvironment

def test_add_measure_helper():
    
    # CartPole (discrete)
    env = CartPoleEnvironment()
    value = get_default_value(env)
    value.setup()
    print(value)
    print(value.value_network)
    print(value.value_network.network)
    print(value.value_optimizer.optimizer)
    qvalue = get_default_qvalue(env)
    qvalue.setup()
    print(qvalue)
    print(qvalue.qvalue_network)
    print(qvalue.qvalue_network.network)
    print(qvalue.qvalue_optimizer.optimizer)
    policy = get_default_policy(env)
    policy.setup()
    print(policy)
    print(policy.policy_network)
    print(policy.policy_network.network)
    print(policy.policy_optimizer.optimizer)

    # Pendulum (continuous)
    env = PendulumEnvironment()
    value = get_default_value(env)
    value.setup()
    # print(value)
    # print(value.value_network)
    # print(value.value_network.network)
    # print(value.value_optimizer.optimizer)
    qvalue = get_default_qvalue(env)
    qvalue.setup()
    # print(qvalue)
    # print(qvalue.qvalue_network)
    # print(qvalue.qvalue_network.network)
    # print(qvalue.qvalue_optimizer.optimizer)
    policy = get_default_policy(env)
    policy.setup()
    # print(policy)
    # print(policy.policy_network)
    # print(policy.policy_network.network)
    # print(policy.policy_optimizer.optimizer)

def main():
    test_add_measure_helper()

if __name__ == "__main__":
    main()