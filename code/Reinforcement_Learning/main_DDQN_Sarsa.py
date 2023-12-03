import gym_super_mario_bros
import pygame
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

from utils.enviroment import *
from utils.agents.MarioDDQN_Sarsa_Agent import *
from utils.setup_env import *
from tqdm import tqdm
import time


def make_env(enviroment):
    enviroment = MaxAndSkipEnv(enviroment)
    enviroment = ProcessFrame84(enviroment)
    enviroment = ImageToPyTorch(enviroment)
    enviroment = BufferWrapper(enviroment, 4)
    enviroment = PixelNormalization(enviroment)
    return JoypadSpace(enviroment, RIGHT_ONLY)


def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]


def run(training_mode, pretrained, double_dqn, num_episodes, exploration_max, sarsa):
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
                     pretrained=pretrained,
                     sarsa=sarsa)

    # Restart the enviroment for each episode
    num_episodes = num_episodes
    enviroment.reset()

    total_rewards = []
    if training_mode and pretrained:
        with open("models/DDQN/total_rewards.pkl", 'rb') as f:
            total_rewards = pickle.load(f)

    if not training_mode:
        init_pygame()

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
        start_time = time.time()
        # num_state = 0
        while True:
            # generate_images_mario(enviroment, ep_num, num_state)
            # num_state += 1
            if not training_mode:
                show_state(enviroment, ep_num)
            action = agent.act(state)
            steps += 1

            state_next, _, terminal, info = enviroment.step(int(action[0]))

            if info["x_pos"] != tmp_info["x_pos"]:
                start_time = time.time()

            custom_reward, tmp_info = custom_rewards(info, tmp_info)

            end_time = time.time()
            if end_time - start_time > 15:
                custom_reward -= CUSTOM_REWARDS["death"]
                terminal = True

            total_reward += custom_reward

            state_next = torch.Tensor([state_next])
            custom_reward = torch.tensor([custom_reward]).unsqueeze(0)

            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, custom_reward, state_next, terminal)

                # Sarsa-network or Q-network
                agent.experience_replay() if not sarsa else agent.experience_replay_sarsa()

            state = state_next

            if terminal:
                break

        total_rewards.append(total_reward)

        if ep_num != 0 and ep_num % 100 == 0:
            print("\nEpisode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1],
                                                                       np.mean(total_rewards)))
        num_episodes += 1
        print(
            "Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1], np.mean(total_rewards)))

    if not training_mode:
        pygame.quit()

    # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
    if training_mode and not sarsa:
        with open("models/DQN/ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("models/DQN/num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("models/DQN/total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
        if agent.double_dqn:
            torch.save(agent.local_net.state_dict(), "models/DQN/DQN1.pt")
            torch.save(agent.target_net.state_dict(), "models/DQN/DQN2.pt")
        else:
            torch.save(agent.dqn.state_dict(), "models/DQN/DQN.pt")
        torch.save(agent.STATE_MEM, "models/DDN/STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "models/DDN/ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "models/DQN/REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, "models/DQN/STATE2_MEM.pt")
        torch.save(agent.DONE_MEM, "models/DQN/DONE_MEM.pt")

    elif training_mode and sarsa:
        with open("sarsa/models/DDN_Sarsa/ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("sarsa/models/DDN_Sarsa/num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("sarsa/models/DDN_Sarsa/total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
        if agent.double_dqn:
            torch.save(agent.local_net.state_dict(), "sarsa/models/DDN_Sarsa/DQN1.pt")
            torch.save(agent.target_net.state_dict(), "sarsa/models/DDN_Sarsa/DQN2.pt")
        else:
            torch.save(agent.dqn.state_dict(), "sarsa/models/DDN_Sarsa/DQN.pt")
        torch.save(agent.STATE_MEM, "sarsa/models/DDN_Sarsa/STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "sarsa/models/DDN_Sarsa/ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "sarsa/models/DDN_Sarsa/REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, "sarsa/models/DDN_Sarsa/STATE2_MEM.pt")
        torch.save(agent.DONE_MEM, "sarsa/models/DDN_Sarsa/DONE_MEM.pt")

    enviroment.close()


if __name__ == "__main__":
    # For training
    run(training_mode=True, pretrained=False, double_dqn=True, num_episodes=10000, exploration_max=1, sarsa=True)

    # For Testing
    run(training_mode=False, pretrained=True, double_dqn=True, num_episodes=20, exploration_max=0.05, sarsa=True)

    # Generate Gif Image
    # generate_gif(image_folder='img', output_gif='demo/ddn_sarsa_victory.gif', file_extension='.png', fps=15)
