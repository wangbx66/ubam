import logging
import json

import numpy as np

class wowah_env(object):
    def __init__(self, ale, agent, num_episodes, epoch_length, test_length,
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
        self.frame_length = 50
        self.buffer_count = 0
        self.screen_buffer = np.empty((self.buffer_length, self.height, self.width), dtype=np.uint8)
        self.terminal_lol = False # Most recent episode ended on a loss of life (unsubscribe of wow)

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for episode in range(1, self.num_episodes + 1):
            self.run_episode(episode, self.test_interval, testing=False)
            self.agent.finish_epoch(epoch)
            with open(self.agent.exp_dir + '/network_file_' + str(episode) + '.pkl', 'w') as fw:
                cPickle.dump(self.agent.network, fw, -1)

            if self.test_episodes > 0:
                self.agent.start_testing()
                self.terminal_lol = False
                logging.info('testing at episode {0}'.format(episode))
                self.run_episode(self.test_episodes, testing=True)
                self.agent.finish_testing(epoch)

    def run_episode(self, episode, max_steps, testing):
        if not testing:
            for num_steps in range(max_steps):
                action, accuracy = self.agent.step()
                evaluation += accuracy
            logging.info('{0} episode: {1} evaluation'.format(episode, evaluation))
            self.agent.end_episode(evaluation, terminal)
        else:
            for num_steps in range(max_steps):
                action, accuracy = self.agent.step()
                evaluator += accuracy