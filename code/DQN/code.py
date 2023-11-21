import pygame
import torch
import torch.nn as nn
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm
import pickle
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym
import collections
import cv2
import pylab as pl
import time
import numpy as np
import matplotlib.pyplot as plt

"""
 MaxAndSkipEnv è un wrapper che modifica il comportamento dell'ambiente Gym.
 Esso restituisce solo ogni skip-esimo frame durante la chiamata del metodo step, 
 consentendo un'approssimazione più rapida dell'azione in situazioni in cui i frame consecutivi possono essere simili.
"""


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


"""
 ProcessFrame84 è un wrapper per l'ambiente Gym che elabora le osservazioni dell'ambiente grezze riducendole a una
  risoluzione di 84x84 pixel e convertendole in scala di grigi.
"""


class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """

    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    """
    process(frame): Metodo statico che prende un frame grezzo come input 
    e lo elabora per ridurlo a una risoluzione di 84x84 pixel. 
    Se il frame ha una dimensione di 240x256x3, lo converte in un array numpy 
    a virgola mobile e quindi esegue una conversione in scala di grigi pesata (luminance).
     Successivamente, ridimensiona l'immagine a 84x110 pixel e ritaglia la parte centrale a 84x84 pixel.
      Restituisce l'immagine elaborata come array numpy con una forma di (84, 84, 1) e tipo di dato np.uint8.
    """

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


"""
ImageToPyTorch(gym.ObservationWrapper): Questo wrapper modifica le osservazioni 
dell'ambiente per adattarle a PyTorch. Nel costruttore, si aggiorna lo spazio delle osservazioni per 
avere dimensioni (old_shape[-1], old_shape[0], old_shape[1]) e tipo di dato np.float32. 
Il metodo observation muove gli assi dell'osservazione in modo che l'ordine degli assi diventi (channel, height, width). 
Questo è comune in PyTorch.
"""


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


"""
ScaledFloatFrame(gym.ObservationWrapper): Questo wrapper normalizza i valori 
dei pixel dell'osservazione nell'intervallo da 0 a 1. 
Il metodo observation prende l'osservazione e restituisce un array numpy normalizzato dividendo ogni valore per 255.0.
"""


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


"""
BufferWrapper(gym.ObservationWrapper): Questo wrapper crea un buffer di osservazioni per tener traccia degli ultimi 
n_steps frame. Nel costruttore, si inizializza lo spazio delle osservazioni con il nuovo spazio in cui ogni osservazione
è ripetuta per n_steps. Il metodo reset azzera il buffer e restituisce l'osservazione iniziale dell'ambiente. 
Il metodo observation aggiorna il buffer spostando gli elementi e inserendo la nuova osservazione. Restituisce 
il buffer aggiornato.
"""


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class Q_Agent():
    """
    Il costruttore inizializza le variabili di stato dell'agente.
    state_a_dict è un dizionario che mappa uno stato a una matrice di valori Q per azioni,
    exploreP è la probabilità di esplorazione iniziale,
    obs_vec è un vettore di osservazioni,
    gamma è il fattore di sconto temporale,
    e alpha è il tasso di apprendimento.
    """

    def __init__(self):
        """ initializing the class"""
        self.state_a_dict = {}
        self.exploreP = 1
        self.obs_vec = []
        self.gamma = 0.99
        self.alpha = 0.01

    """
    obs_to_state(self, obs): Questo metodo converte un'osservazione in uno stato.
    Se l'osservazione è già presente in obs_vec, 
    restituisce l'indice corrispondente. 
    Altrimenti, aggiunge l'osservazione a obs_vec e restituisce il nuovo indice.
    """

    def obs_to_state(self, obs):
        state = -1
        for i in range(len(self.obs_vec)):
            if np.array_equal(obs, self.obs_vec[i]):
                state = i
                break
        if state == -1:
            state = len(self.obs_vec)
            self.obs_vec.append(obs)
        return state

    """
    take_action(self, state): 
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

    def take_action(self, state):
        Q_a = self.get_Qval(state)
        if (np.random.rand() > self.exploreP):
            """ exploitation"""
            action = np.argmax(Q_a)
        else:
            """ exploration"""
            action = env.action_space.sample()

        self.exploreP *= 0.99
        return action

    """
    get_Qval(self, state): Restituisce i valori Q corrispondenti a uno stato. 
    Se lo stato non è presente in state_a_dict, inizializza casualmente i valori Q.
    """

    def get_Qval(self, state):
        if (state not in self.state_a_dict):
            self.state_a_dict[state] = np.random.rand(5, 1)
        return self.state_a_dict[state]

    """
    update_Qval(self, action, state, reward, next_state, terminal): 
    Aggiorna i valori Q in base all'equazione di aggiornamento di Q-learning. 
    Calcola il target temporale TD_target e il calcolo dell'errore temporale 
    td_error per aggiornare i valori Q associati all'azione e allo stato corrente.
    """

    def update_Qval(self, action, state, reward, next_state, terminal):
        if terminal:
            TD_target = reward
        else:
            TD_target = reward + self.gamma * np.amax(self.get_Qval(next_state))

        td_error = TD_target - self.get_Qval(state)[action]
        self.state_a_dict[state][action] += self.alpha * td_error


def make_env(env):
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return JoypadSpace(env, RIGHT_ONLY)


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((240, 256))
    pygame.display.set_caption("Super Mario Bros")
    return screen


def show_state(env, ep=0, info=""):
    screen = pygame.display.get_surface()
    image = env.render(mode='rgb_array')
    image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    screen.blit(image, (0, 0))
    pygame.display.flip()
    pygame.display.set_caption(f"Episode: {ep} {info}")
    pygame.time.delay(50)  # Aggiungi un ritardo per rallentare la visualizzazione


def agent_training(num_episodes):
    rewards = []

    for i_episode in range(num_episodes):
        obs = env.reset()
        state = Mario.obs_to_state(obs)
        episode_reward = 0
        while True:
            action = Mario.take_action(state)
            next_obs, reward, terminal, _ = env.step(action)
            episode_reward += reward

            next_state = Mario.obs_to_state(next_obs)

            Mario.update_Qval(action, state, reward, next_state, terminal)
            state = next_state

            if terminal:
                break
        rewards.append(episode_reward)
        print("Total reward after episode {} is {}".format(i_episode + 1, episode_reward))

        # Saving the reward array every 10 episodes
        if i_episode % 10 == 0:
            np.save('rewards.npy', rewards)


def agent_testing(num_episodes):
    total_rewards = []

    init_pygame()
    for i_episode in range(num_episodes):
        obs = env.reset()
        state = Mario.obs_to_state(obs)
        episode_reward = 0

        while True:
            # Sfrutta il modello addestrato senza esplorazione
            # perché l'obiettivo è valutare le prestazioni
            # del modello addestrato, non esplorare nuove azioni.
            show_state(env, i_episode)

            action = np.argmax(Mario.get_Qval(state))
            next_obs, reward, terminal, _ = env.step(action)
            episode_reward += reward

            next_state = Mario.obs_to_state(next_obs)
            state = next_state

            if terminal:
                break

        total_rewards.append(episode_reward)
        print(f"Total reward after testing episode {i_episode + 1} is {episode_reward}")

    pygame.quit()
    average_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} testing episodes: {average_reward}")


if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = make_env(env)  # Wraps the environment so that frames are grayscale
    obs = env.reset()

    env.observation_space
    env.action_space

    # Imposta a True se vuoi utilizzare un agente già addestrato
    use_trained_agent = False

    if use_trained_agent:
        # Carica i valori Q appresi durante l'addestramento
        with open('trained_q_values.pkl', 'rb') as f:
            trained_q_values = pickle.load(f)

        Mario = Q_Agent()
        Mario.state_a_dict = trained_q_values
    else:
        # Crea un nuovo agente non addestrato
        Mario = Q_Agent()
        agent_training(num_episodes=10)

        with open('trained_q_values.pkl', 'wb') as f:
            pickle.dump(Mario.state_a_dict, f)

    agent_testing(num_episodes=10)

    # Plotting graph
    rewards = np.load('rewards.npy')
    plt.title("Episodes trained vs. Average Rewards (per 5 eps)")
    plt.plot(np.convolve(rewards, np.ones((5,)) / 5, mode="valid").tolist())
    plt.show()



'''def agent_testing(num_test_episodes):
    # Imposta l'esplorazione a zero per la fase di test
    Mario.exploreP = 0

    init_pygame()
    for i_episode in range(num_test_episodes):
        obs = env.reset()
        state = Mario.obs_to_state(obs)
        episode_reward = 0

        while True:
            show_state(env, i_episode)
            action = np.argmax(Mario.get_Qval(state))  # Usa sempre l'azione con il massimo valore Q
            next_obs, reward, terminal, _ = env.step(action)
            episode_reward += reward

            next_state = Mario.obs_to_state(next_obs)

            state = next_state

            if terminal:
                break

        print("Total reward after test episode {} is {}".format(i_episode + 1, episode_reward))
    pygame.quit()'''
