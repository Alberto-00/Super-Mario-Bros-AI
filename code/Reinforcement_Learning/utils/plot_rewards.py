import numpy as np
import matplotlib.pyplot as plt
import os

path = "../models/DoubleQL/"
path_sarsa = "../sarsa/models/QL/"
rewards = np.load(os.path.abspath(path + "rewards.npy"))
# rewards = np.load(os.path.abspath(path + "total_rewards.pkl"), allow_pickle=True)

# Prendiamo solo le prime 1000 ricompense
first_1000_rewards = rewards[:1000]

# Calcoliamo la media delle prime 1000 ricompense
average_rewards = np.mean(first_1000_rewards)

# Visualizza il grafico delle prime 1000 ricompense
plt.title("Episodes trained vs. Avarage Rewards (per 1000 eps)")
plt.plot(np.convolve(first_1000_rewards, np.ones((100,))/100, mode="valid").tolist())
plt.xlabel("Episodes")
plt.ylabel("Avarage Rewards")
plt.show()
