#### Properties ####

# env
props_env = {
    "CartPole":
    {
        "is_discrete_state_space": False,
        "is_discrete_action_space": True,
        "input_shape": 4,
        "output_shape": 2
    },
    "GridWorld":
    {
        "is_discrete_state_space": True,
        "is_discrete_action_space": True,
        "input_shape": 25,
        "output_shape": 4
    },
}

# agent
props_agent = {
    "DQN":
    {
        "is_model_based": False,
        "is_model_free": True,
        "is_value_based": True,
        "is_policy_based": False,
        "is_on_policy": False,
        "is_off_policy": True,
        "is_deterministic_policy": False,
        "is_online": False
    },
    "SAC":
    {
        "is_model_based": False,
        "is_model_free": True,
        "is_value_based": True,
        "is_policy_based": True,
        "is_on_policy": False,
        "is_off_policy": True,
        "is_deterministic_policy": False,
        "is_online": False
    },
    "Qlearning":
    {
        "is_model_based": False,
        "is_model_free": True,
        "is_value_based": True,
        "is_policy_based": True,
        "is_on_policy": False,
        "is_off_policy": True,
        "is_deterministic_policy": False,
        "is_online": True
    }
}

# actor
props_actor = {

}

# critic
props_critic = {

}

# brain
# props_brain = {
# }

# value
props_value = {

}

# policy
props_policy = {

}

# network
props_net = {

}