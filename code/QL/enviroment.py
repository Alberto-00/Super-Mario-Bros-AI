import gym
import collections
import cv2
import numpy as np


class MaxAndSkipEnv(gym.Wrapper):
    """
    MaxAndSkipEnv è un wrapper che modifica il comportamento dell'ambiente Gym. Esso restituisce solo ogni skip-esimo
    frame durante la chiamata del metodo step, consentendo un'approssimazione più rapida dell'azione in situazioni in
    cui i frame consecutivi possono essere simili.
    """
    def __init__(self, enviroment=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(enviroment)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        for _ in range(self._skip):
            observation, reward, done, info = self.env.step(action)
            self._obs_buffer.append(observation)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        observation = self.env.reset()
        self._obs_buffer.append(observation)
        return observation


class ProcessFrame84(gym.ObservationWrapper):
    """
    ProcessFrame84 è un wrapper per l'ambiente Gym che elabora le osservazioni dell'ambiente grezze riducendole a una
    risoluzione di 84x84 pixel e convertendole in scala di grigi.

    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """

    def __init__(self, enviroment=None):
        super(ProcessFrame84, self).__init__(enviroment)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, observation):
        return ProcessFrame84.process(observation)

    @staticmethod
    def process(frame):
        """
           process(frame): Metodo statico che prende un frame grezzo come input
           e lo elabora per ridurlo a una risoluzione di 84x84 pixel.
           Se il frame ha una dimensione di 240x256x3, lo converte in un array numpy
           a virgola mobile e quindi esegue una conversione in scala di grigi pesata (luminance).
            Successivamente, ridimensiona l'immagine a 84x110 pixel e ritaglia la parte centrale a 84x84 pixel.
             Restituisce l'immagine elaborata come array numpy con una forma di (84, 84, 1) e tipo di dato np.uint8.
        """
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    """
    ImageToPyTorch(gym.ObservationWrapper): Questo wrapper modifica le osservazioni dell'ambiente per adattarle a
    PyTorch. Nel costruttore, si aggiorna lo spazio delle osservazioni per avere dimensioni (old_shape[-1],
    old_shape[0], old_shape[1]) e tipo di dato np.float32. Il metodo observation muove gli assi dell'osservazione in
    modo che l'ordine degli assi diventi (channel, height, width). Questo è comune in PyTorch.
    """
    def __init__(self, enviroment):
        super(ImageToPyTorch, self).__init__(enviroment)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    ScaledFloatFrame(gym.ObservationWrapper): Questo wrapper normalizza i valori dei pixel dell'osservazione
    nell'intervallo da 0 a 1. Il metodo observation prende l'osservazione e restituisce un array numpy normalizzato
    dividendo ogni valore per 255.0.

    Normalize pixel values in frame --> 0 to 1
    """
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    """
    BufferWrapper(gym.ObservationWrapper): Questo wrapper crea un buffer di osservazioni per tener traccia degli
    ultimi n_steps frame. Nel costruttore, si inizializza lo spazio delle osservazioni con il nuovo spazio in cui
    ogni osservazione è ripetuta per n_steps. Il metodo reset azzera il buffer e restituisce l'osservazione iniziale
    dell'ambiente. Il metodo observation aggiorna il buffer spostando gli elementi e inserendo la nuova osservazione.
    Restituisce il buffer aggiornato.
    """
    def __init__(self, enviroment, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(enviroment)
        self.buffer = None
        self.dtype = dtype
        old_space = enviroment.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer
