#### Utility ####

import itertools
import numpy as np
import torch


# function -> vector
def fun2vec(f, *domains):
    v = np.zeros(domains)
    for args in itertools.product(*[list(range(domain)) for domain in domains]):
        v[args] = f(*args)
    return v


# vector -> function
def vec2fun(v, dim):
    f = lambda *indices: v[indices[:dim]]
    return f


# map function
def map_over_trajectory(f, state_traj, action_traj):
    n = min(len(state_traj), len(action_traj))
    tmp = f(state_traj)
    return torch.cat([tmp[[_n], [action_traj[_n]]] for _n in range(n)], axis=0)


# actual return
def actual_return(trajectory, gamma, start=0, end=np.inf):

    assert(start <= end)
    c = 0.
    g = 1.

    for t, (_, _, r, _) in enumerate(trajectory):
        if (start <= t and t < end):
            c += g * r
            g *= gamma
    
    return c


# actual return of all time steps
def actual_returns(trajectory, gamma):

    T = len(trajectory)
    c = 0.
    cs = [0 for _ in range(T)]

    r = sars2r(trajectory)
    for t in reversed(range(T)):
        if (t < T-1):
            cs[t] = r[t] + gamma * cs[t+1]
        else:
            cs[t] = r[t]
    
    return cs

