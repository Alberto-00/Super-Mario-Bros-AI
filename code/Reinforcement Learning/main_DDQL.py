import os

import pygame
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

from utils.enviroment import *
from utils.agents.MarioDDQLAgent import *
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
    enviroment = PixelNormalization(enviroment)
    return JoypadSpace(enviroment, RIGHT_ONLY)


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((240, 256))
    pygame.display.set_caption("Super Mario Bros")
    return screen


def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]


def run(training_mode, pretrained, double_dqn, num_episodes, exploration_max):
    enviroment = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    enviroment = make_env(enviroment)  # Wraps the environment so that frames are grayscale
    observation_space = enviroment.observation_space.shape
    action_space = enviroment.action_space.n
    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=30000,
                     batch_size=32,
                     gamma=0.90,
                     lr=0.00025,
                     dropout=0.2,
                     exploration_max=exploration_max,
                     exploration_min=0.02,
                     exploration_decay=0.99,
                     double_dqn=double_dqn,
                     pretrained=pretrained)

    # Restart the enviroment for each episode
    num_episodes = num_episodes
    enviroment.reset()

    total_rewards = []
    if training_mode and pretrained:
        with open("/models/DDQL/total_rewards.pkl", 'rb') as f:
            total_rewards = pickle.load(f)

    for ep_num in tqdm(range(num_episodes)):
        state = enviroment.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        tmp_info = {
            'coins': 0, 'flag_get': False,
            'life': 2, 'status': 'small',
            'TimeLimit.truncated': True,
            'x_pos': 40, 'score': 0,
            'time': 400
        }
        while True:
            if not training_mode:
                show_state(enviroment, ep_num)
            action = agent.act(state)
            steps += 1

            state_next, _, terminal, info = enviroment.step(int(action[0]))
            custom_reward, tmp_info = custom_rewards(info, tmp_info)

            total_reward += custom_reward
            state_next = torch.Tensor([state_next])
            custom_reward = torch.tensor([custom_reward]).unsqueeze(0)

            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, custom_reward, state_next, terminal)
                agent.experience_replay()

            state = state_next
            if terminal:
                break

        total_rewards.append(total_reward)

        if ep_num != 0 and ep_num % 100 == 0:
            print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1],
                                                                     np.mean(total_rewards)))
        num_episodes += 1

        print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1], np.mean(total_rewards)))

    # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
    if training_mode:
        with open("/models/DDQL/ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("/models/DDQL/num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("/models/DDQL/total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
        if agent.double_dqn:
            torch.save(agent.local_net.state_dict(), "/models/DDQL/DQN1.pt")
            torch.save(agent.target_net.state_dict(), "/models/DDQL/DQN2.pt")
        else:
            torch.save(agent.dqn.state_dict(), "/models/DDQL/DQN.pt")
        torch.save(agent.STATE_MEM, "/models/DDQL/STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "/models/DDQL/ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "/models/DDQL/REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, "/models/DDQL/STATE2_MEM.pt")
        torch.save(agent.DONE_MEM, "/models/DDQL/DONE_MEM.pt")

    enviroment.close()


if __name__ == "__main__":
    # For training
    run(training_mode=True, pretrained=False, double_dqn=True, num_episodes=3181, exploration_max=1)

    # For Testing
    run(training_mode=False, pretrained=True, double_dqn=True, num_episodes=1, exploration_max=0.05)
