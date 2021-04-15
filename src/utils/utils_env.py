#### Utility ####

import numpy as np
import torch
from PIL import Image


# {(s,a,r,s_next)} -> {(s,a,r,s_next,a_next)}
# action on a terminal state can be chosen arbitrarily
def sars2sarsa(history):
    history_sarsa = []
    t = 0; T = len(history)
    for t in range(T):
        s, a, r, s_next = history[t]
        _, a_next, _, _ = history[(t+1)%T]
        history_sarsa.append((s,a,r,s_next,a_next))
    return history_sarsa


# {(s, a, r, s_next)} -> ({s}, {a}, {r}, {s_next})
def unzip_trajectory(trajectory):

    state_traj, action_traj, reward_traj, state_next_traj = zip(*trajectory)
    state_traj = torch.from_numpy(np.array(state_traj).astype(np.float32))
    action_traj = torch.from_numpy(np.array(action_traj).astype(np.float32))
    reward_traj = torch.from_numpy(np.array(reward_traj).astype(np.float32))
    state_next_traj = torch.from_numpy(np.array(state_next_traj).astype(np.float32))

    return state_traj, action_traj, reward_traj, state_next_traj


# ({s}, {a}, {r}, {s_next}) -> {(s, a, r, s_next)}
def zip_trajectory(state_traj, action_traj, reward_traj, state_next_traj):
    
    state_traj = state_traj.detach().numpy()
    action_traj = action_traj.detach().numpy()
    reward_traj = reward_traj.detach().numpy()
    state_next_traj = state_next_traj.detach().numpy()

    return list(zip(state_traj, action_traj, reward_traj, state_next_traj))


# preprocessing for Breakout
def preprocess(image):
    image = image[:,:,0]
    image = np.array(Image.fromarray(image).resize((84, 110), resample=2))[110-84-8:110-8,:] / 256.
    image = torch.tensor(image, dtype=float)
    return image