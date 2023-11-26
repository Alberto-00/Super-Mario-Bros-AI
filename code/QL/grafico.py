import numpy as np
import matplotlib.pyplot as plt
import os

rewards = np.load(os.path.abspath("model_1/rewards.npy"))
plt.title("Episodes trained vs. Average Rewards (per 5 eps)")
plt.plot(rewards)
plt.show()