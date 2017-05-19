import numpy as np


class BaseEnvironment(object):
    def get_initial_state(self):
        """
        Sets the environment to its initial state.
        :return: the initial state
        """
        raise NotImplementedError()

    def next(self, action):
        """
        Appies the current action to the environment.
        :param action: one hot vector.
        :return: (observation, reward, is_terminal) tuple
        """
        raise NotImplementedError()

    def get_legal_actions(self):
        """
        Get the set of indices of legal actions
        :return: a numpy array of the indices of legal actions
        """
        raise NotImplementedError()

    def get_noop(self):
        """
        Gets the no-op action, to be used with self.next
        :return: the action
        """
        raise NotImplementedError()

    def on_new_frame(self, frame):
        """
        Called whenever a new frame is available.
        :param frame: raw frame
        """
        pass


class FramePool(object):

    def __init__(self, frame_pool, operation):
        self.frame_pool = frame_pool
        self.frame_pool_index = 0
        self.frames_in_pool = frame_pool.shape[0]
        self.operation = operation

    def new_frame(self, frame):
        self.frame_pool[self.frame_pool_index] = frame
        self.frame_pool_index = (self.frame_pool_index + 1) % self.frames_in_pool

    def get_processed_frame(self):
        return self.operation(self.frame_pool)


class ObservationPool(object):

    def __init__(self, observation_pool):
        self.observation_pool = observation_pool
        self.pool_size = observation_pool.shape[-1]
        self.permutation = [self.__shift(list(range(self.pool_size)), i) for i in range(self.pool_size)]
        self.current_observation_index = 0

    def new_observation(self, observation):
        self.observation_pool[:, :, self.current_observation_index] = observation
        self.current_observation_index = (self.current_observation_index + 1) % self.pool_size

    def get_pooled_observations(self):
        return np.copy(self.observation_pool[:, :, self.permutation[self.current_observation_index]])

    def __shift(self, seq, n):
        n = n % len(seq)
        return seq[n:]+seq[:n]

