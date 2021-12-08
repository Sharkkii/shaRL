#### RL Dataset ####

from torch.utils.data import Dataset

class RLDataset(Dataset):

    def __init__(
        self,
        n_capacity
    ):
        self.dataset = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": []
        }
        self.n_capacity = n_capacity

    def __len__(
        self
    ):
        l_state = len()
        l_action = len(self.dataset["action"])
        l_reward = len(self.dataset["reward"])
        l_next_state = len(self.dataset["next_state"])
        assert(l_state == l_action and l_state == l_reward and l_state == l_next_state)
        return l_state

    def __getitem__(
        self,
        index
    ):
        state = self.dataset["state"][index]
        action = self.dataset["action"][index]
        reward = self.dataset["reward"][index]
        next_state = self.dataset["next_state"][index]
        return (state, action, reward, next_state)

    def load(
        self
    ):
        state = self.dataset["state"]
        action = self.dataset["action"]
        reward = self.dataset["reward"]
        next_state = self.dataset["next_state"]
        history = RLDataset.zip(state, action, reward, next_state)
        return history
    
    def save(
        self,
        history
    ):
        state, action, reward, next_state = RLDataset.unzip(history)
        self.dataset["state"].extend(state)
        self.dataset["action"].extend(action)
        self.dataset["reward"].extend(reward)
        self.dataset["next_state"].extend(next_state)
        self._discard()

    def _discard(
        self
    ):
        for key in self.dataset.keys():
            self.dataset[key] = self.dataset[key][- self.n_capacity:]

    # {(s, a, r, s_next)} -> ({s}, {a}, {r}, {s_next})
    def unzip(
        history
    ):
        state_traj, action_traj, reward_traj, state_next_traj = zip(*history)
        return state_traj, action_traj, reward_traj, state_next_traj

    # ({s}, {a}, {r}, {s_next}) -> {(s, a, r, s_next)}
    def zip(
        state_traj,
        action_traj,
        reward_traj,
        state_next_traj
    ):
        history = list(zip(state_traj, action_traj, reward_traj, state_next_traj))
        return history
