import numpy as np
from ale_python_interface import ALEInterface
from scipy.misc import imresize
import random
from environment import BaseEnvironment, FramePool,ObservationPool

IMG_SIZE_X = 84
IMG_SIZE_Y = 84
NR_IMAGES = 4
ACTION_REPEAT = 4
MAX_START_WAIT = 30
FRAMES_IN_POOL = 2


class AtariEmulator(BaseEnvironment):
    def __init__(self, actor_id, args):
        self.ale = ALEInterface()
        self.ale.setInt(b"random_seed", args.random_seed * (actor_id +1))
        # For fuller control on explicit action repeat (>= ALE 0.5.0)
        self.ale.setFloat(b"repeat_action_probability", 0.0)
        # Disable frame_skip and color_averaging
        # See: http://is.gd/tYzVpj
        self.ale.setInt(b"frame_skip", 1)
        self.ale.setBool(b"color_averaging", False)
        full_rom_path = args.rom_path + "/" + args.game + ".bin"
        self.ale.loadROM(str.encode(full_rom_path))
        self.legal_actions = self.ale.getMinimalActionSet()
        self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.lives = self.ale.lives()

        self.random_start = args.random_start
        self.single_life_episodes = args.single_life_episodes
        self.call_on_new_frame = args.visualize

        # Processed historcal frames that will be fed in to the network 
        # (i.e., four 84x84 images)
        self.observation_pool = ObservationPool(np.zeros((IMG_SIZE_X, IMG_SIZE_Y, NR_IMAGES), dtype=np.uint8))
        self.rgb_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.gray_screen = np.zeros((self.screen_height, self.screen_width,1), dtype=np.uint8)
        self.frame_pool = FramePool(np.empty((2, self.screen_height,self.screen_width), dtype=np.uint8),
                                    self.__process_frame_pool)

    def get_legal_actions(self):
        return self.legal_actions

    def __get_screen_image(self):
        """
        Get the current frame luminance
        :return: the current frame
        """
        self.ale.getScreenGrayscale(self.gray_screen)
        if self.call_on_new_frame:
            self.ale.getScreenRGB(self.rgb_screen)
            self.on_new_frame(self.rgb_screen)
        return np.squeeze(self.gray_screen)

    def on_new_frame(self, frame):
        pass

    def __new_game(self):
        """ Restart game """
        self.ale.reset_game()
        self.lives = self.ale.lives()
        if self.random_start:
            wait = random.randint(0, MAX_START_WAIT)
            for _ in range(wait):
                self.ale.act(self.legal_actions[0])

    def __process_frame_pool(self, frame_pool):
        """ Preprocess frame pool """
        
        img = np.amax(frame_pool, axis=0)
        img = imresize(img, (84, 84), interp='nearest')
        img = img.astype(np.uint8)
        return img

    def __action_repeat(self, a, times=ACTION_REPEAT):
        """ Repeat action and grab screen into frame pool """
        reward = 0
        for i in range(times - FRAMES_IN_POOL):
            reward += self.ale.act(self.legal_actions[a])
        # Only need to add the last FRAMES_IN_POOL frames to the frame pool
        for i in range(FRAMES_IN_POOL):
            reward += self.ale.act(self.legal_actions[a])
            self.frame_pool.new_frame(self.__get_screen_image())
        return reward

    def get_initial_state(self):
        """ Get the initial state """
        self.__new_game()
        for step in range(NR_IMAGES):
            _ = self.__action_repeat(0)
            self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        if self.__is_terminal():
            raise Exception('This should never happen.')
        return self.observation_pool.get_pooled_observations()

    def next(self, action):
        """ Get the next state, reward, and game over signal """

        reward = self.__action_repeat(np.argmax(action))
        self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        terminal = self.__is_terminal()
        self.lives = self.ale.lives()
        observation = self.observation_pool.get_pooled_observations()
        return observation, reward, terminal
            
    def __is_terminal(self):
        if self.single_life_episodes:
            return self.__is_over() or (self.lives > self.ale.lives())
        else:
            return self.__is_over()

    def __is_over(self):
        return self.ale.game_over()
