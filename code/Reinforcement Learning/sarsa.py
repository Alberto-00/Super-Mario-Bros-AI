import os

import pygame
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import pickle
from gym_super_mario_bros.actions import RIGHT_ONLY
import matplotlib.pyplot as plt

from utils.enviroment import *
from utils.MarioQLAgent import MarioQLAgent
from tqdm import tqdm
import time


CUSTOM_REWARDS = {
    "time": -0.1,  # per second that passes by
    "death": -100.,  # mario dies
    "extra_life": 100.,  # mario gets an extra life, which includes getting 100th coin
    "mushroom": 20.,  # mario eats a mushroom to become big
    "flower": 25.,  # mario eats a flower
    "mushroom_hit": -10.,  # mario gets hit while big
    "flower_hit": -15.,  # mario gets hit while fire mario
    "coin": 15.,  # mario gets a coin
    "score": 15.,  # mario hit enemies
    "victory": 1000  # mario win
}


def show_state(enviroment, ep=0, info=""):
    screen = pygame.display.get_surface()
    image = enviroment.render(mode='rgb_array')
    image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    screen.blit(image, (0, 0))
    pygame.display.flip()
    pygame.display.set_caption(f"Episode: {ep} {info}")
    pygame.time.delay(50)  # Aggiungi un ritardo per rallentare la visualizzazione


def custom_rewards(name, tmp_info):
    reward = 0

    # detect score
    if tmp_info['score'] != name['score']:
        reward += CUSTOM_REWARDS['score']

    # detect x_pos
    if tmp_info['x_pos'] != name['x_pos']:
        reward += name['x_pos'] - tmp_info['x_pos']

    # detect time
    if tmp_info['time'] != name['time']:
        reward += CUSTOM_REWARDS['time']

    # detect if finished
    if tmp_info['flag_get'] != name['flag_get'] and name['flag_get']:
        reward += CUSTOM_REWARDS['victory']

    # detect deaths
    if 'TimeLimit.truncated' in name:
        reward += CUSTOM_REWARDS["death"]

    # detect extra lives
    if tmp_info['life'] != name['life'] and name["life"] > 2:
        reward += CUSTOM_REWARDS['extra_life']

    # detect getting a coin
    if tmp_info['coins'] != name['coins']:
        reward += CUSTOM_REWARDS['coin']

        if name["coins"] > 6:
            reward += 500

    # detect if mario ate a mushroom, ate a flower, or got hit without dying
    if tmp_info['status'] != name['status']:
        # 2 - fire mario. only achieved if eating flower while super mario
        # 1 - super mario. only achieved if eating mushroom while small mario
        # 0 - small mario. only achieved if hit while super mario or fire mario. if hit while small mario, death.

        # if small value was sent, you got hit when you were big
        if tmp_info['status'] == 'tall' and name['status'] == 'small':
            reward += CUSTOM_REWARDS['mushroom_hit']

        # or worse, you got hit when you were a flower
        elif tmp_info['status'] == 'fireball' and name['status'] == 'tall':
            reward += CUSTOM_REWARDS['flower_hit']

        # ate a flower (assuming was still super mario. if eating flower while small mario, mario only becomes super
        # mario so this value would be a value of 1, and be caught in the value == 1 checks)
        elif name['status'] == 'fireball':
            reward += CUSTOM_REWARDS['flower']

        # if currently super mario, only need to check if this is from eating mushroom. if hit while fire mario,
        # goes back to small mario
        elif name['status'] == 'tall':
            reward += CUSTOM_REWARDS['mushroom']

    return reward, name


def make_env(enviroment):
    enviroment = MaxAndSkipEnv(enviroment)
    enviroment = ProcessFrame84(enviroment)
    enviroment = ImageToPyTorch(enviroment)
    enviroment = BufferWrapper(enviroment, 4)
    enviroment = ScaledFloatFrame(enviroment)
    return JoypadSpace(enviroment, RIGHT_ONLY)


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((240, 256))
    pygame.display.set_caption("Super Mario Bros")
    return screen


def agent_training(num_episodes, total_rewards, mario_agent, enviroment):
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
                np.save(os.path.abspath("models/QL/rewards.npy"), np.array(total_rewards))
                with open(os.path.abspath("models/QL/model.pkl"), 'wb') as file:
                    pickle.dump(agent_mario.state_a_dict, file)

                print("\nRewards and model are saved.\n")


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

            action = mario_agent.take_action_sarsa(state, next_action=None)

            while True:
                next_obs, _, terminal, info = enviroment.step(action)

                # Utilizza la funzione di ricompensa personalizzata
                custom_reward, tmp_info = custom_rewards(info, tmp_info)

                episode_reward += custom_reward
                next_state = mario_agent.obs_to_state(next_obs)

                # Seleziona l'azione successiva usando SARSA
                next_action = mario_agent.take_action_sarsa(next_state, next_action=None)

                # Aggiorna i valori Q utilizzando SARSA
                mario_agent.update_qval(action, state, custom_reward, next_state, next_action, terminal)

                state = next_state
                action = next_action

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
                np.save(os.path.abspath("models/rewards-sampleModel.npy"), np.array(total_rewards))

                with open(os.path.abspath("models/agent_mario-sampleModel.pkl"), 'wb') as file:
                    pickle.dump(mario_agent.state_a_dict, file)

                print("\nModel and rewards are saved.\n")


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

    agent_mario = MarioQLAgent(env)

    # Imposta a True se vuoi utilizzare un agente già addestrato
    use_trained_agent = False

    # Imposta a True se vuoi effettuare la fase di training
    training = True

    if use_trained_agent:
        # Carica i valori Q appresi e le rewards durante l'addestramento
        with open(os.path.abspath("models/QL/model.pkl"), 'rb') as f:
            trained_q_values = pickle.load(f)

        rewards = np.load(os.path.abspath("models/QL/rewards.npy"))
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
    rewards = np.load(os.path.abspath("models/QL/rewards.npy"))
    plt.title("Episodes trained vs. Average Rewards (per 5 eps)")
    plt.plot(np.convolve(rewards, np.ones((5,)) / 5, mode="valid").tolist())
    plt.show()
