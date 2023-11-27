import numpy as np


class DoubleQLAgent:

    def __init__(self, env):
        """ initializing the class"""
        self.state_a_dict = {}
        self.Q_target = {}
        self.copy_steps = 10
        self.exploreP = 1
        self.env = env
        self.obs_vec = []
        self.gamma = 0.99
        self.alpha = 0.01

    def obs_to_state(self, obs):
        state = -1
        for i in range(len(self.obs_vec)):
            if (obs == self.obs_vec[i]).all():
                state = i
                break
        if state == -1:
            state = len(self.obs_vec)
            self.obs_vec.append(obs)
        return state

    def take_action(self, state):
        q_a = self.get_qval(state)
        if np.random.rand() > self.exploreP:
            """ exploitation"""
            action = np.argmax(q_a)
        else:
            """ exploration"""
            action = self.env.action_space.sample()

        # Assicurati che l'azione sia all'interno del range valido (da 0 a 4)
        action = np.clip(action, 0, 4)

        self.exploreP *= 0.99
        return action

    def get_qval(self, state):
        if state not in self.state_a_dict:
            self.state_a_dict[state] = np.random.rand(5, 1)
        return self.state_a_dict[state]

    def get_qtarget(self, state):
        if state not in self.Q_target:
            self.Q_target[state] = np.random.rand(5, 1)
        return self.Q_target[state]

    def update_qval(self, state, action, reward, next_state, terminal):
        if terminal:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.amax(self.get_qtarget(next_state))

        td_error = td_target - self.get_qval(state)[action]
        self.state_a_dict[state][action] += self.alpha * td_error

    def copy(self):
        self.Q_target = self.state_a_dict.copy()

    def sarsa_update(self, state, action, reward, next_state, next_action, terminal):
        if terminal:
            target = reward
        else:
            next_action_q_value = self.get_qval(next_state)[next_action]
            target = reward + self.gamma * next_action_q_value

        current_q_value = self.get_qval(state)[action]

        # SARSA update
        updated_q_value = current_q_value + self.alpha * (target - current_q_value)

        # Update Q-values
        self.state_a_dict[state][action] = updated_q_value

