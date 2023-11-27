import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from Reinforcement_Learning.utils.enviroment import *
from Reinforcement_Learning.utils.agents.MarioDDQLAgent import *
from Reinforcement_Learning.utils.setup_env import *
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

def sarsa_update(agent, state, action, reward, next_state, next_action, terminal):
    # Calcola l'obiettivo SARSA
    if terminal:
        target = reward
    else:
        next_action_q_value = agent.local_net(next_state).gather(1, next_action.long())
        target = reward + agent.gamma * next_action_q_value

    # Calcola l'attuale valore Q per l'azione corrente
    current = agent.local_net(state).gather(1, action.long())

    # Converte target e current in tensori PyTorch e abilita il gradiente
    target = torch.tensor(target, dtype=torch.float32, requires_grad=True).to(agent.device)
    current = torch.tensor(current, dtype=torch.float32, requires_grad=True).to(agent.device)

    # Calcola la perdita e esegui l'aggiornamento del modello
    loss = agent.l1(current, target)
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()



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
        start_time = time.time()
        action = agent.act(state)

        while True:
            if not training_mode:
                show_state(enviroment, ep_num)

            next_state, _, terminal, info = enviroment.step(int(action[0]))
            custom_reward, tmp_info = custom_rewards(info, tmp_info)
            total_reward += custom_reward
            next_state = torch.Tensor([next_state])

            next_action = agent.act(next_state)

            if training_mode:
                sarsa_update(agent, state, action, custom_reward, next_state, next_action, terminal)

            state = next_state
            action = next_action

            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if terminal:
                break

        total_rewards.append(total_reward)

        if ep_num != 0 and ep_num % 100 == 0:
            print("\nEpisode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1],
                                                                       np.mean(total_rewards)))
        num_episodes += 1
        print(
            "\nEpisode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1], np.mean(total_rewards)))

    # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
    if training_mode:
        with open("models/DDQL/ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("models/DDQL/num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("models/DDQL/total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
        if agent.double_dqn:
            torch.save(agent.local_net.state_dict(), "models/DDQL/DQN1.pt")
            torch.save(agent.target_net.state_dict(), "models/DDQL/DQN2.pt")
        else:
            torch.save(agent.dqn.state_dict(), "models/DDQL/DQN.pt")
        torch.save(agent.STATE_MEM, "models/DDQL/STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "models/DDQL/ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "models/DDQL/REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, "models/DDQL/STATE2_MEM.pt")
        torch.save(agent.DONE_MEM, "models/DDQL/DONE_MEM.pt")

    enviroment.close()


if __name__ == "__main__":
    # For training
    run(training_mode=True, pretrained=False, double_dqn=True, num_episodes=3181, exploration_max=1)

    # For Testing
    run(training_mode=False, pretrained=True, double_dqn=True, num_episodes=1, exploration_max=0.05)
