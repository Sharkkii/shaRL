#### Data ####

class BaseData:
    pass

class SA(BaseData):

    def __init__(
        self,
        state,
        action
    ):
        self.state = state
        self.action = action

class SARS(BaseData):

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

class SARSA(BaseData):

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

class SAG(BaseData):

    def __init__(
        self,
        state,
        action,
        goal
    ):
        self.state = state
        self.action = action
        self.goal = goal
