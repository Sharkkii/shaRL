#### Data ####

import torch

class BaseData:
    
    @classmethod
    def random_state(
        cls,
        n = 1
    ):
        state_set = [ torch.randn(size = (1,)) for _ in range(n) ]
        return state_set

    @classmethod
    def random_action(
        cls,
        n = 1
    ):
        action_set = [ torch.randint(low = 0, high = 2, size = (1,)) for _ in range(n) ]
        return action_set

    @classmethod
    def random_reward(
        cls,
        n = 1
    ):
        reward_set = [ torch.rand(size = (1,)) for _ in range(n) ]
        return reward_set

    @classmethod
    def random_goal(
        cls,
        n = 1
    ):
        goal_set = [ torch.randn(size = (1,)) for _ in range(n) ]
        return goal_set


class Data(BaseData):
    pass


class SA(Data):

    def __init__(
        self,
        state,
        action
    ):
        self.state = state
        self.action = action

    @classmethod
    def random(
        cls,
        n = 1
    ):
        state_set = super().random_state(n = n)
        action_set = super().random_action(n = n)
        dataset = [
            SA(state, action) for state, action in zip(state_set, action_set)
        ]
        return dataset


class SARS(Data):

    def __init__(
        self,
        state,
        action,
        reward,
        next_state
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    @classmethod
    def random(
        cls,
        n = 1
    ):
        state_set = super().random_state(n = n)
        action_set = super().random_action(n = n)
        reward_set = super().random_reward(n = n)
        next_state_set = super().random_state(n = n)
        dataset = [
            SARS(state, action, reward, next_state) for state, action, reward, next_state in zip(state_set, action_set, reward_set, next_state_set)
        ]
        return dataset


class SARSA(Data):

    def __init__(
        self,
        state,
        action,
        reward,
        next_state,
        next_action
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.next_action = next_action

    @classmethod
    def random(
        cls,
        n = 1
    ):
        state_set = super().random_state(n = n)
        action_set = super().random_action(n = n)
        reward_set = super().random_reward(n = n)
        next_state_set = super().random_state(n = n)
        next_action_set = super().random_action(n = n)
        dataset = [
            SARSA(state, action, reward, next_state, next_action) for state, action, reward, next_state, next_action in zip(state_set, action_set, reward_set, next_state_set, next_action_set)
        ]
        return dataset


class SGASG(Data):

    def __init__(
        self,
        state,
        goal,
        action,
        next_state,
        next_goal
    ):
        self.state = state
        self.goal = goal
        self.action = action
        self.next_state = next_state
        self.next_goal = next_goal

    @classmethod
    def random(
        cls,
        n = 1
    ):
        state_set = super().random_state(n = n)
        goal_set = super().random_goal(n = n)
        action_set = super().random_action(n = n)
        next_state_set = super().random_state(n = n)
        next_goal_set = super().random_goal(n = n)
        dataset = [
            SGASG(state, goal, action, next_state, next_goal) for state, goal, action, next_state, next_goal in zip(state_set, goal_set, action_set, next_state_set, next_goal_set)
        ]
        return dataset


class SGARSG(Data):

    def __init__(
        self,
        state,
        goal,
        action,
        reward,
        next_state,
        next_goal
    ):
        self.state = state
        self.goal = goal
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.next_goal = next_goal

    @classmethod
    def random(
        cls,
        n = 1
    ):
        state_set = super().random_state(n = n)
        goal_set = super().random_goal(n = n)
        action_set = super().random_action(n = n)
        reward_set = super().random_reward(n = n)
        next_state_set = super().random_state(n = n)
        next_goal_set = super().random_goal(n = n)
        dataset = [
            SGARSG(state, goal, action, reward, next_state, next_goal) for state, goal, action, reward, next_state, next_goal in zip(state_set, goal_set, action_set, reward_set, next_state_set, next_goal_set)
        ]
        return dataset
