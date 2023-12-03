import torch
import torch.nn as nn
import random
import pickle
import numpy as np


class DQNSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """

    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay, double_dqn, pretrained, sarsa):

        # Define DQN Layers = Deep Sarsa Layer
        self.state_space = state_space
        self.action_space = action_space
        self.double_dqn = double_dqn
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sarsa = sarsa

        # Double DQN network = DD Sarsa network
        if self.double_dqn:
            self.local_net = DQNSolver(state_space, action_space).to(self.device)
            self.target_net = DQNSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                if not self.sarsa:
                    self.local_net.load_state_dict(torch.load("models/DQN/DQN1.pt",
                                                              map_location=torch.device(self.device)))
                    self.target_net.load_state_dict(torch.load("models/DQN/DQN2.pt",
                                                               map_location=torch.device(self.device)))
                else:
                    self.local_net.load_state_dict(torch.load("sarsa/models/DDN_Sarsa/DQN1.pt",
                                                              map_location=torch.device(self.device)))
                    self.target_net.load_state_dict(torch.load("sarsa/models/DDN_Sarsa/DQN2.pt",
                                                               map_location=torch.device(self.device)))

            self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
            self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
            self.step = 0
        # DQN network  = Deep Sarsa Layer
        else:
            self.dqn = DQNSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                if not sarsa:
                    self.dqn.load_state_dict(torch.load("models/DQN/DQN.pt", map_location=torch.device(self.device)))
                else:
                    self.dqn.load_state_dict(torch.load("sarsa/models/DDN_Sarsa/DQN.pt", map_location=torch.device(self.device)))
            self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            if not sarsa:
                self.STATE_MEM = torch.load("models/DQN/STATE_MEM.pt")
                self.ACTION_MEM = torch.load("models/DQN/ACTION_MEM.pt")
                self.REWARD_MEM = torch.load("models/DQN/REWARD_MEM.pt")
                self.STATE2_MEM = torch.load("models/DQN/STATE2_MEM.pt")
                self.DONE_MEM = torch.load("models/DQN/DONE_MEM.pt")
                with open("models/DQN/ending_position.pkl", 'rb') as f:
                    self.ending_position = pickle.load(f)
                with open("models/DQN/num_in_queue.pkl", 'rb') as f:
                    self.num_in_queue = pickle.load(f)
            else:
                self.STATE_MEM = torch.load("sarsa/models/DDN_Sarsa/STATE_MEM.pt")
                self.ACTION_MEM = torch.load("sarsa/models/DDN_Sarsa/ACTION_MEM.pt")
                self.REWARD_MEM = torch.load("sarsa/models/DDN_Sarsa/REWARD_MEM.pt")
                self.STATE2_MEM = torch.load("sarsa/models/DDN_Sarsa/STATE2_MEM.pt")
                self.DONE_MEM = torch.load("sarsa/models/DDN_Sarsa/DONE_MEM.pt")
                with open("sarsa/models/DDN_Sarsa/ending_position.pkl", 'rb') as f:
                    self.ending_position = pickle.load(f)
                with open("sarsa/models/DDN_Sarsa/num_in_queue.pkl", 'rb') as f:
                    self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        state = self.STATE_MEM[idx]
        action = self.ACTION_MEM[idx]
        reward = self.REWARD_MEM[idx]
        state2 = self.STATE2_MEM[idx]
        done = self.DONE_MEM[idx]
        return state, action, reward, state2, done

    def act(self, state):
        """Epsilon-greedy action"""
        if self.double_dqn:
            self.step += 1
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        if self.double_dqn:
            # Local net is used for the policy
            return torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def copy_model(self):
        """Copy local net weights into target net for DDQN network"""
        self.target_net.load_state_dict(self.local_net.state_dict())

    def experience_replay(self):
        """Use the double Q-update or Q-update equations to update the network weights"""
        if self.double_dqn and self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        # Sample a batch of experiences
        state, action, reward, state2, done = self.batch_experiences()
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        state2 = state2.to(self.device)
        done = done.to(self.device)

        self.optimizer.zero_grad()
        if self.double_dqn:
            # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
            target = reward + torch.mul((self.gamma * self.target_net(state2).max(1).values.unsqueeze(1)), 1 - done)

            current = self.local_net(state).gather(1, action.long())  # Local net approximation of Q-value
        else:
            # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
            target = reward + torch.mul((self.gamma * self.dqn(state2).max(1).values.unsqueeze(1)), 1 - done)

            current = self.dqn(state).gather(1, action.long())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

    def experience_replay_sarsa(self):
        """Use the double Sarsa-update or Sarsa-update equations to update the network weights"""
        if self.double_dqn and self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        # Sample a batch of experiences
        state, action, reward, state2, done = self.batch_experiences()
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        state2 = state2.to(self.device)
        done = done.to(self.device)

        self.optimizer.zero_grad()

        # double_sarsa
        if self.double_dqn:
            # Il target di Double SARSA è Q*(S, A) <- r + γ Q_target(S', A')
            next_action = self.local_net(state2).max(1).indices.unsqueeze(1)
            target = reward + torch.mul(self.gamma * self.target_net(state2).gather(1, next_action), 1 - done)
            current = self.local_net(state).gather(1, action.long())
        else:
            # Il target di SARSA è Q*(S, A) <- r + γ Q(S', A')
            next_action = self.dqn(state2).argmax(1).unsqueeze(1)
            target = reward + torch.mul(self.gamma * self.dqn(state2).gather(1, next_action), 1 - done)
            current = self.dqn(state).gather(1, action.long())

        loss = self.l1(current, target)
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
