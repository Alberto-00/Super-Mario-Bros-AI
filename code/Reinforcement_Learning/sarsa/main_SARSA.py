import os

import pygame
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import pickle
from gym_super_mario_bros.actions import RIGHT_ONLY

from Reinforcement_Learning.utils.enviroment import *
from Reinforcement_Learning.utils.agents.MarioQLAgent import MarioQLAgent
from Reinforcement_Learning.utils.setup_env import *
from tqdm import tqdm
import time


def make_env(enviroment):
    enviroment = MaxAndSkipEnv(enviroment)
    enviroment = ProcessFrame84(enviroment)
    enviroment = ImageToPyTorch(enviroment)
    enviroment = BufferWrapper(enviroment, 4)
    enviroment = ScaledFloatFrame(enviroment)
    return JoypadSpace(enviroment, RIGHT_ONLY)


def agent_training_sarsa(num_episodes, total_rewards, mario_agent, enviroment):
    with tqdm(total=num_episodes, desc="Training Episodes") as progress_bar:
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
            action = mario_agent.take_action(state)
            while True:
                next_obs, _, terminal, info = enviroment.step(action)

                if info["x_pos"] != tmp_info["x_pos"]:
                    start_time = time.time()

                custom_reward, tmp_info = custom_rewards(info, tmp_info)

                end_time = time.time()
                if end_time - start_time > 15:
                    custom_reward -= CUSTOM_REWARDS["death"]
                    terminal = True

                next_state = mario_agent.obs_to_state(next_obs)
                next_action = mario_agent.take_action(next_state)

                mario_agent.update_qval_sarsa(action, state, custom_reward, next_state, next_action, terminal)

                state = next_state
                action = next_action

                episode_reward += custom_reward

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
                np.save(os.path.abspath("models/Sarsa/rewards.npy"), np.array(total_rewards))
                with open(os.path.abspath("models/Sarsa/model.pkl"), 'wb') as file:
                    pickle.dump(agent_mario.state_a_dict, file)

                print("\nRewards and model are saved.\n")


def agent_testing_sarsa(num_episodes, mario_agent, enviroment):
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

        action = np.argmax(mario_agent.get_qval(state))
        while True:
            show_state(enviroment, i_episode)
            next_obs, _, terminal, info = enviroment.step(action)

            custom_reward, tmp_info = custom_rewards(info, tmp_info)
            episode_reward += custom_reward

            next_state = mario_agent.obs_to_state(next_obs)
            # Scegli un'azione successiva usando il modello addestrato
            next_action = np.argmax(mario_agent.get_qval(next_state))
            state = next_state
            action = next_action

            if terminal:
                break

        total_rewards.append(episode_reward)
        print(f"Total reward after testing episode {i_episode + 1} is {episode_reward}")

    pygame.quit()


if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = make_env(env)  # Wraps the environment so that frames are grayscale
    obs = env.reset()

    agent_mario = MarioQLAgent(env)

    # Imposta a True se vuoi utilizzare un agente già addestrato
    use_trained_agent = False

    # Imposta a True se vuoi effettuare la fase di training
    training = True

    if use_trained_agent:
        # Carica i valori Q appresi e le rewards durante l'addestramento
        with open(os.path.abspath("models/Sarsa/model.pkl"), 'rb') as f:
            trained_q_values = pickle.load(f)

        rewards = np.load(os.path.abspath("models/Sarsa/rewards.npy"))
        agent_mario.state_a_dict = trained_q_values

        if training:
            agent_training_sarsa(num_episodes=1000, total_rewards=rewards, mario_agent=agent_mario, enviroment=env)
        agent_testing_sarsa(num_episodes=5, mario_agent=agent_mario, enviroment=env)

    else:
        if training:
            rewards = []
            agent_training_sarsa(num_episodes=1000, total_rewards=rewards, mario_agent=agent_mario, enviroment=env)
        agent_testing_sarsa(num_episodes=1, mario_agent=agent_mario, enviroment=env)
