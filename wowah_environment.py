import logging
import json

import numpy as np

class wowah_env(object):
    def __init__(self, ale, agent, width, height, num_episodes, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng):
        with open('data/zonesjson.txt') as fp:
            self.zones = json.loads(fp.readline())
            self.lvls = json.loads(fp.readline())
        self.agent = agent
        self.test_interval = test_interval
        self.num_episodes = 1000
        self.action_set = set(zones)
        self.rng = rng

        # The followings for test use only
        self.test_episodes = 1000
        self.frame_skip = 4
        self.buffer_length = 50
        self.buffer_count = 0
        self.screen_buffer = np.empty((self.buffer_length, self.height, self.width), dtype=np.uint8)
        self.terminal_lol = False # Most recent episode ended on a loss of life (unsubscribe of wow)
        

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for episode in range(1, self.num_episodes + 1):
            self.run_episode(self.test_interval, testing=False)
            self.agent.finish_epoch(epoch)
            with open(self.agent.exp_dir + '/network_file_' + str(episode) + '.pkl', 'w') as fw:
                cPickle.dump(self.agent.network, fw, -1)

            if self.test_episodes > 0:
                self.agent.start_testing()
                self.terminal_lol = False
                logging.info('testing at episode {0}'.format(episode))
                self.run_episode(self.test_episodes, testing=True)
                self.agent.finish_testing(epoch)

    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        self.terminal_lol = False # Make sure each epoch starts with a reset.
        steps_left = num_steps
        while steps_left > 0:
            prefix = "testing" if testing else "training"
            logging.info(prefix + " epoch: " + str(epoch) + " steps_left: " +
                         str(steps_left))
            _, num_steps = self.run_episode(steps_left, testing)

            steps_left -= num_steps


    def _init_episode(self):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""

        if not self.terminal_lol or self.ale.game_over():
            self.ale.reset_game()

            if self.max_start_nullops > 0:
                random_actions = self.rng.randint(0, self.max_start_nullops+1)
                for _ in range(random_actions):
                    self._act(0) # Null action

        # Make sure the screen buffer is filled at the beginning of
        # each episode...
        self._act(0)
        self._act(0)


    def _act(self, action):
        """Perform the indicated action for a single frame, return the
        resulting reward and store the resulting screen image in the
        buffer

        """
        reward = self.ale.act(action)
        index = self.buffer_count % self.buffer_length

        self.ale.getScreenGrayscale(self.screen_buffer[index, ...])

        self.buffer_count += 1
        return reward

    def _step(self, action):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        reward = 0
        for _ in range(self.frame_skip):
            reward += self._act(action)

        return reward

    def run_episode(self, max_steps, testing):
        self._init_episode()
        
        start_lives = self.ale.lives()

        action = self.agent.start_episode(self.get_observation())
        num_steps = 0
        while True:
            reward = self._step(self.min_action_set[action])
            self.terminal_lol = (self.death_ends_episode and not testing and
                                 self.ale.lives() < start_lives)
            terminal = self.ale.game_over() or self.terminal_lol
            num_steps += 1

            if terminal or num_steps >= max_steps:
                self.agent.end_episode(reward, terminal)
                break

            action = self.agent.step(reward, self.get_observation())
        return terminal, num_steps


    def get_observation(self):
        """ Resize and merge the previous two screen images """

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        return self.resize_image(max_image)

    def resize_image(self, image):
        """ Appropriately resize a single image """

        if self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(image,
                                 (self.resized_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
            cropped = resized[crop_y_cutoff:
                              crop_y_cutoff + self.resized_height, :]

            return cropped
        elif self.resize_method == 'scale':
            return cv2.resize(image,
                              (self.resized_width, self.resized_height),
                              interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('Unrecognized image resize method.')

