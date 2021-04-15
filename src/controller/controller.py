#### Controller ####

import numpy as np
import torch

import sys
sys.path.append("../src")
from props import props_env, props_agent

class Controller:

    def __init__(
        self,
        environment,
        agent,
        config={
            "env_name": "Default",
            "agent_name": "Default"
        }
    ):

        self.env = environment
        self.agent = agent
        self.config = config

        env_name = config["env_name"]
        agent_name = config["agent_name"]
        self.props_env = props_env[env_name]
        self.props_agent = props_agent[agent_name]


    def reset(
        self
    ):
        self.env.reset()
        self.agent.reset()
    
    def fit(
        self,
        n_epoch=1,
        n_sample=1,
        n_sample_start=1,
        n_eval=1,
        n_eval_interval=1,
        env_step=1,
        gradient_step=1,
    ):
        is_model_based = self.props_agent["is_model_based"]
        is_model_free = self.props_agent["is_model_free"]
        is_value_based = self.props_agent["is_value_based"]
        is_policy_based = self.props_agent["is_policy_based"]
        is_on_policy = self.props_agent["is_on_policy"]
        is_off_policy = self.props_agent["is_off_policy"]
        is_online = self.props_agent["is_online"]
        assert(is_on_policy == (not is_off_policy))

        # online
        if (is_online):
            env_step = 1
            is_model_based = False
            is_on_policy = True
            is_off_policy = False

        # TODO: -> agent.setup (see props.py)
        is_discrete_state_space = self.props_env["is_discrete_state_space"]
        is_discrete_action_space = self.props_env["is_discrete_action_space"]
        is_deterministic_policy = self.props_agent["is_deterministic_policy"]

        assert(n_sample <= n_sample_start <= self.agent.memory.capacity)

        for epoch in range(n_epoch):

            self.agent.setup_every_epoch(
                epoch=epoch,
                n_epoch=n_epoch
            )

            # train

            # interact & prepare training data
            trajs_env = []
            trajs_model = []

            for _ in range(env_step):
                
                # interact w/ environment
                if (is_model_free):
                    trajs_env = self.agent.interact_with_env(
                        self.env,
                        is_discrete_action_space=is_discrete_action_space,
                        is_deterministic_policy=is_deterministic_policy,
                        n_times=1
                    )

                # interact w/ model    
                if (is_model_based):
                    trajs_model = self.agent.interact_with_model(
                        is_discrete_action_space=is_discrete_action_space,
                        is_deterministic_policy=is_deterministic_policy,
                        n_times=1
                    )
            
            trajs = trajs_env + trajs_model

            if (is_off_policy):
                self.agent.save_trajs(trajs)
                # guard
                if (self.agent.memory.count < n_sample_start):
                    continue
                trajs = self.agent.load_trajs(n_sample=n_sample)

            # optimize policy
            # FIXME: J_v, J_q, J_pi will be overwritten
            for _ in range(gradient_step):

                # update value function (critic)
                J_v, J_q = self.agent.update_critic(trajs, n_times=1)
                # update policy (actor)
                J_pi = self.agent.update_actor(trajs, n_times=1)

                # learn dynamics
                if (is_model_based):
                    J_m = self.agent.update_model(n_times=1)
                else:
                    J_m = None

            # TODO: learn something if needed
            # self.agent.update_every_epoch(
            #     epoch=epoch,
            #     n_epoch=n_epoch
            # )

            # evaluate
            print("%d" % epoch, end="\r", flush=True)
            if ((epoch+1) % n_eval_interval == 0):
                print(epoch+1, end=" | ")
                result = self.evaluate(n_eval)
                if (J_v is not None):
                    print("J_v: %2.4f" % J_v, end=" | ")
                else:
                    print(J_v, end=" | ")
                if (J_q is not None):
                    print("J_q: %2.4f" % J_q, end=" | ")
                else:
                    print(J_q, end=" | ")
                if (J_pi is not None):
                    print("J_pi: %2.4f" % J_pi, end=" | ")
                else:
                    print(J_pi,  end=" | ")
                print("duration: %d" % result[0], end=" ")
                print("total reward: %f" % result[1], end=" ")
                print()

    def evaluate(
        self,
        n_eval=1
    ):
        result = self.agent.evaluate(self.env, n_eval)
        return result
        




    