import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import pickle
from gym_super_mario_bros.actions import RIGHT_ONLY
import matplotlib.pyplot as plt

from utils.enviroment import *
from utils.agents.MarioDoubleQLAgent import DoubleQLAgent
from utils.setup_env import *
from tqdm import tqdm
import time


def make_env(enviroment):
    enviroment = MaxAndSkipEnv(enviroment)
    enviroment = ProcessFrame84(enviroment)
    enviroment = ImageToPyTorch(enviroment)
    enviroment = BufferWrapper(enviroment, 4)
    enviroment = ScaledFloatFrame(enviroment)
    return JoypadSpace(enviroment, RIGHT_ONLY)


def agent_training(num_episodes, total_rewards, mario_agent, enviroment):
    with tqdm(total=num_episodes, desc="Training Episodes") as progress_bar:
        num_steps = 0
        for i_episode in range(num_episodes):
            observation = enviroment.reset()
            state = mario_agent.obs_to_state(observation)
            episode_reward = 0
            tmp_info = {
                'coins': 0, 'flag_get': False,
                'life': 2, 'status': 'small',
                'TimeLimit.truncated': True,
                'x_pos': 40, 'score': 0,
                'time': 400
            }

            start_time = time.time()
            while True:
                action = mario_agent.take_action(state)
                next_obs, _, terminal, info = enviroment.step(action)

                if info["x_pos"] != tmp_info["x_pos"]:
                    start_time = time.time()

                # Utilizza la funzione di ricompensa personalizzata
                custom_reward, tmp_info = custom_rewards(info, tmp_info)

                episode_reward += custom_reward

                end_time = time.time()
                if end_time - start_time > 15:
                    custom_reward -= CUSTOM_REWARDS["death"]
                    terminal = True

                num_steps += 1
                if num_steps % mario_agent.copy_steps == 0:
                    mario_agent.copy()

                next_state = mario_agent.obs_to_state(next_obs)

                mario_agent.update_qval(action, state, custom_reward, next_state, terminal)
                state = next_state

                if terminal:
                    break

            if isinstance(total_rewards, np.ndarray):
                total_rewards = np.append(total_rewards, episode_reward)
            else:
                total_rewards.append(episode_reward)

            progress_bar.update(1)
            progress_bar.set_postfix({'Reward': episode_reward})

            # Saving the reward array and agent every 10 episodes
            if i_episode % 10 == 0:
                np.save(os.path.abspath("models/DoubleQL/rewards.npy"), np.array(total_rewards))
                with open(os.path.abspath("models/DoubleQL/model.pkl"), 'wb') as file:
                    pickle.dump(agent_mario.state_a_dict, file)

                print("\nRewards and model are saved.\n")


def agent_testing(num_episodes, mario_agent, enviroment):
    total_rewards = []
    init_pygame()

    for i_episode in range(num_episodes):
        observation = enviroment.reset()
        state = mario_agent.obs_to_state(observation)
        episode_reward = 0
        tmp_info = {
            'coins': 0, 'flag_get': False,
            'life': 2, 'status': 'small',
            'TimeLimit.truncated': True,
            'x_pos': 40, 'score': 0,
            'time': 400
        }

        while True:
            # Sfrutta il modello addestrato senza esplorazione
            # perché l'obiettivo è valutare le prestazioni
            # del modello addestrato, non esplorare nuove azioni.
            show_state(enviroment, i_episode)
            action = np.argmax(mario_agent.get_qval(state))
            next_obs, _, terminal, info = enviroment.step(action)

            custom_reward, tmp_info = custom_rewards(info, tmp_info)

            episode_reward += custom_reward

            next_state = mario_agent.obs_to_state(next_obs)
            state = next_state

            if terminal:
                break

        total_rewards.append(episode_reward)
        print(f"Total reward after testing episode {i_episode + 1} is {episode_reward}")

    pygame.quit()


if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = make_env(env)  # Wraps the environment so that frames are grayscale
    obs = env.reset()

    agent_mario = DoubleQLAgent(env)

    # Imposta a True se vuoi utilizzare un agente già addestrato
    use_trained_agent = False

    # Imposta a True se vuoi effettuare la fase di training
    training = True

    if use_trained_agent:
        # Carica i valori Q appresi e le rewards durante l'addestramento
        with open(os.path.abspath("models/DoubleQL/model.pkl"), 'rb') as f:
            trained_q_values = pickle.load(f)

        rewards = np.load(os.path.abspath("models/DoubleQL/rewards.npy"))
        agent_mario.state_a_dict = trained_q_values

        if training:
            agent_training(num_episodes=5000, total_rewards=rewards, mario_agent=agent_mario, enviroment=env)
        agent_testing(num_episodes=5, mario_agent=agent_mario, enviroment=env)

    else:
        if training:
            rewards = []
            agent_training(num_episodes=5000, total_rewards=rewards, mario_agent=agent_mario, enviroment=env)
        agent_testing(num_episodes=1, mario_agent=agent_mario, enviroment=env)

    # Plotting graph
    rewards = np.load(os.path.abspath("models/DoubleQL/rewards.npy"))
    plt.title("Episodes trained vs. Average Rewards (per 5 eps)")
    plt.plot(np.convolve(rewards, np.ones((5,)) / 5, mode="valid").tolist())
    plt.show()
