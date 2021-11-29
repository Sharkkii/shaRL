# shaRL: Scalable, Highly Adaptable Reinforcement Learning Framework

## Description

**shaRL** is a reinforcement learning framework entirely written by Python. The concept of shaRL is "Reinforcement Learning as Building Blocks"; shaRL is attempted to help users to construct RL algorithms. Not only using existing RL algorithms but also implementing new ones can be accomplished with much less effort!

## Characterstics

- Scalable
shaRL is greatly modularized; it is built from smaller submodules/components following the principle of object-oriented programming. This philosophy is well suited for RL because many RL algorithms share the whole structure and constituents in one algorithm can be reused in other algorithms. When we develop a new RL alrogithm, what the developers/users only have to do is to implement the novel part of the algorithm to be introduced.

- Comprehensive
RL is not so easy to understand especially for beginners. shaRL will be very helpful for such people to overview how RL algorithm goes, because the top-level module only has abstract expressions or calls subroutines to put details behind.

## Modules
Here are higher-level common modules used in shaRL framework. Click the link to know the details:

- [Controller](https://github.com/Sharkkii/shaRL/tree/develop/src/controller)
`Controller` is the top-level module of shaRL. This manages a pair of `Environment` and `Agent`, which are the main components of RL.

- [Environment](https://github.com/Sharkkii/shaRL/tree/develop/src/environment)
`Environment` is a higher-level module that manages the world where `Agent` takes actions. Environments used in shaRL are based on [Open AI Gym environment](https://gym.openai.com/envs/#classic_control).

- [Agent](https://github.com/Sharkkii/shaRL/tree/develop/src/agent)
`Agent` is a higher-level module that manages a pair of `Actor` and `Critic`. `Agent` chooses actions according to the policy (wrapped by the actor).

- [Actor](https://github.com/Sharkkii/shaRL/tree/develop/src/actor)
`Actor` is a wrapper module that manages `Policy`. `Actor` can share its knowledge with `Critic`.

- [Critic](https://github.com/Sharkkii/shaRL/tree/develop/src/critic)
`Critic` is a wrapper module that manages `Value`. `Critic` can share its knowledge with `Actor`.

- [Policy](https://github.com/Sharkkii/shaRL/tree/develop/src/policy)
`Policy` determines how the agent moves around in an environment.

- [Value](https://github.com/Sharkkii/shaRL/tree/develop/src/value)
`Value` indicates how good a state / state-action pair is in a given environment. It will provide important information to the agent's policy (through the wrapper module `Critic`).



