# import numpy as np

# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# from src.memory import RLDataset
# from src.memory import RLDataLoader
# from src.environment import CartPoleEnvironment

# def test_add_rldataset():

#     env = CartPoleEnvironment()
#     buffer = RLDataset(
#         min_size = 10,
#         max_size = 10
#     )

#     for _ in range(10):
#         state = env.reset()
#         done = False
#         trajectory = []
#         for _ in range(3):
#             if (done): break
#             action = env.action_space.sample()
#             next_state, reward, done, _ = env.step(action)
#             trajectory.append((state, action, reward, next_state))
#             state = next_state
#         buffer.save(trajectory)
#         print(np.array(buffer.dataset["state"]))

#     print("OK: test_add_rldataset")

# def test_add_rldataloader():

#     env = CartPoleEnvironment()
#     dataset = RLDataset(
#         min_size = 10,
#         max_size = 10
#     )
#     dataloader = RLDataLoader(
#         dataset = dataset,
#         batch_size = 2,
#         shuffle = False
#     )

#     for _ in range(10):
#         state = env.reset()
#         done = False
#         trajectory = []
#         for _ in range(3):
#             if (done): break
#             action = env.action_space.sample()
#             next_state, reward, done, _ = env.step(action)
#             trajectory.append((state, action, reward, next_state))
#             state = next_state
#         dataloader.save(trajectory)
    
#     state, _, _, _ = RLDataset.unzip(dataloader.load())
#     print(state)
    
#     for history in dataloader:
#         state, _, _, _ = history
#         print(state)

#     print("OK: test_add_rldataloader")

# def main():
#     test_add_rldataset()
#     test_add_rldataloader()

# if __name__ == "__main__":
#     main()