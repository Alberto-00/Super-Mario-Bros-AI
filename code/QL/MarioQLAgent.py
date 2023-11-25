import torch
import numpy as np


class MarioQLAgent:
    """
    Il costruttore inizializza le variabili di stato dell'agente.
    state_a_dict è un dizionario che mappa uno stato a una matrice di valori Q per azioni,
    exploreP è la probabilità di esplorazione iniziale,
    obs_vec è un vettore di osservazioni,
    gamma è il fattore di sconto temporale,
    e alpha è il tasso di apprendimento.
    """

    def __init__(self, env):
        """ initializing the class"""
        self.state_a_dict = {}
        self.exploreP = 1
        self.obs_vec = []
        self.gamma = 0.99  # si può modificare
        self.alpha = 0.01  # si può modificare
        self.explore_decay = 0.99  # si può modificare
        self.explore_min = 0.02  # si può modificare
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def obs_to_state(self, observation):
        """
        obs_to_state(self, obs): Questo metodo converte un'osservazione in uno stato.
        Se l'osservazione è già presente in obs_vec,
        restituisce l'indice corrispondente.
        Altrimenti, aggiunge l'osservazione a obs_vec e restituisce il nuovo indice.
        """
        state = -1
        for i in range(len(self.obs_vec)):
            if np.array_equal(observation, self.obs_vec[i]):
                state = i
                break
        if state == -1:
            state = len(self.obs_vec)
            self.obs_vec.append(observation)
        return state

    def take_action(self, state):
        """
        Questo metodo seleziona un'azione in base alla politica dell'agente.
        Se un valore casuale è maggiore di exploreP, l'agente esegue l'azione corrispondente al massimo valore Q.
        Altrimenti, l'agente esegue un'azione casuale.
        La probabilità di esplorazione exploreP diminuisce nel tempo.

        La politica dell'agente è controllata dalla condizione np.random.rand() > self.exploreP. Se questa condizione
        è vera, l'agente esegue l'azione corrispondente al massimo valore Q (exploitation). Altrimenti, l'agente
        esegue un'azione casuale (exploration). La probabilità di esplorazione exploreP diminuisce nel tempo, poiché
        viene moltiplicata per 0.99 ad ogni chiamata di take_action.

        Quindi, in questo caso, la politica è una combinazione di esplorazione e sfruttamento, dove l'agente inizialmente
        esplora di più (azioni casuali) e gradualmente si orienta verso l'exploitation (sfruttamento dei valori Q appresi).
        """
        if np.random.rand() < self.exploreP:
            # Exploration: choose a random action
            action = self.env.action_space.sample()
        else:
            # Exploitation: choose action with the highest Q-value
            q_values = self.state_a_dict[state]
            max_q_value = np.max(q_values)

            # Add some randomness to the selection among the best actions
            best_actions = [a for a, q in enumerate(q_values) if q == max_q_value]
            action = np.random.choice(best_actions)

        # Decay exploration probability
        self.exploreP *= self.explore_decay
        self.exploreP = max(self.exploreP, int(self.explore_min))

        return action

    def get_qval(self, state):
        """
        Restituisce i valori Q corrispondenti a uno stato.
        Se lo stato non è presente in state_a_dict, inizializza casualmente i valori Q.
        """
        if state not in self.state_a_dict:
            self.state_a_dict[state] = np.random.rand(7, 1)
        return self.state_a_dict[state]


    def update_qval(self, action, state, reward, next_state, terminal):
        """
        Aggiorna i valori Q in base all'equazione di aggiornamento di Q-learning.
        Calcola il target temporale TD_target e il calcolo dell'errore temporale
        td_error per aggiornare i valori Q associati all'azione e allo stato corrente.
        """
        if terminal:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.amax(self.get_qval(next_state))

        td_error = td_target - self.get_qval(state)[action]
        self.state_a_dict[state][action] += self.alpha * td_error

    """def update_qval(self, action, state, reward, next_state, next_action, terminal):
        if terminal:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.get_qval(next_state)[next_action]

        td_error = td_target - self.get_qval(state)[action]
        self.state_a_dict[state][action] += self.alpha * td_error"""






    def take_action_sarsa(self, state, next_action):
        """
        Questo metodo seleziona un'azione in base all'algoritmo SARSA.
        Se un valore casuale è maggiore di exploreP, l'agente esegue l'azione corrispondente al massimo valore Q.
        Altrimenti, l'agente esegue un'azione casuale.
        La probabilità di esplorazione exploreP diminuisce nel tempo.

        La differenza rispetto a take_action è che prende in input anche l'azione successiva (next_action).
        """
        if next_action is not None:
            return next_action

        if np.random.rand() < self.exploreP:
            # Exploration: choose a random action
            action = self.env.action_space.sample()
        else:
            # Exploitation: choose action with the highest Q-value
            if state in self.state_a_dict:
                q_values = self.state_a_dict[state]
                max_q_value = np.max(q_values)

                # Add some randomness to the selection among the best actions
                best_actions = [a for a, q in enumerate(q_values) if q == max_q_value]
                action = np.random.choice(best_actions)
            else:
                # If the state is not in the dictionary, choose a random action
                action = self.env.action_space.sample()

        # Decay exploration probability
        self.exploreP *= self.explore_decay
        self.exploreP = max(self.exploreP, self.explore_min)

        return action