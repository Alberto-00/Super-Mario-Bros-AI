import numpy as np
import matplotlib.pyplot as plt
import os

path_ql = "../models/DDQN/"
path_sarsa = "../sarsa/models/DDN_Sarsa/"

# Carica le ricompense dai file
rewards_ql = np.load(os.path.abspath(path_ql + "total_rewards.pkl"), allow_pickle=True)
rewards_sarsa = np.load(os.path.abspath(path_sarsa + "total_rewards.pkl"), allow_pickle=True)

# Prendiamo solo le prime 1000 ricompense per entrambi i modelli
first_1000_rewards_ql = rewards_ql[:1000]
first_1000_rewards_sarsa = rewards_sarsa[:1000]

# Calcoliamo le medie delle prime 1000 ricompense per entrambi i modelli
average_rewards_ql = np.mean(first_1000_rewards_ql)
average_rewards_sarsa = np.mean(first_1000_rewards_sarsa)

# Visualizza il grafico delle prime 1000 ricompense per entrambi i modelli
plt.title("Episodes trained vs. Average Rewards (per 10000 eps)")
plt.plot(np.convolve(first_1000_rewards_ql, np.ones((100,))/100, mode="valid").tolist(), label="DDQN")
plt.plot(np.convolve(first_1000_rewards_sarsa, np.ones((100,))/100, mode="valid").tolist(), label="DDN SARSA")
plt.xlabel("Episodes")
plt.ylabel("Average Rewards")
plt.legend()
plt.show()
