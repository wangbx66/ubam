import os
import pickle
import json
import time
import logging
from itertools import islice

import numpy as np
import h5py

import sys
sys.setrecursionlimit(10000)

def frames_dep(num_frames=-1, require_expert=False):
    self.num_cat_feature = 4
    self.num_ord_feature = 2
    with open('data/zonesjson.txt') as fp:
        zones = json.loads(fp.readline())
        lvls = json.loads(fp.readline())
    if require_expert:
        fp = open('data/expertsjson.txt')
    else:
        fp = open('data/usersjson.txt')
    users = json.loads(fp.readline())
    k = np.array(users.keys())
    p = np.array(users.values())
    p /= (p - self.frame_length).sum()
    fp.close()
    while True:
        if num_frames > 0:
            num_frames -= 1
            if num_frames == 0:
                break
        user = rng.choice(k, p=p)
        cat_input = np.zeros((self.frame_length, self.num_cat_feature), dtype=np.uint8)
        ord_input = np.zeros((self.frame_length, self.num_ord_feature), dtype=np.float32)
        cat_iutput_prime = np.zeros((self.frame_length, self.num_cat_feature), dtype=np.uint8)
        ord_iutput_prime = np.zeros((self.frame_length, self.num_ord_feature), dtype=np.float32)
        reward = 0
        action = 0
        with open(os.path.join('data/users', user)) as fp:
            start = rng.uniform(low=0, high=users[user] - frame_skip - frame_length)

            for idx, line in enumerate(fp):
                idx_logstats, user, tt, guild, lvl, race, category, zone, seq = line.strip().split(',')
                if idx >= start and idx < start + frame_length:
                    cat_input[idx] = np.array([guild, race, category, zone])
                    ord_input[idx] = np.array([lvl, idx])
                if idx >= start + frame_skip and idx < start + frame_length + frame_skip:
                    cat_input_prime[idx] = np.array([guild, race, category, zone])
                    ord_input_prime[idx] = np.array([lvl, idx])
                if idx == start + frame_length - 1:
                    lvl_in = lvl
                if idx == start + frame_length + frame_skip - 1:
                    lvl_out = lvl
        reward = lvl_out - lvl_in
        action = np.argmax(np.bincount(cat_input_prime[-frame_skip:, -1]))
        action_set = lvls[lvl_in]
        yield (cat_input, ord_input, cat_iutput_prime, ord_input_prime, reward, action, action_set)

def frames(num_frames=10, skip_frames=4, require_expert=False, rng=np.random.RandomState(123456)):
    num_cat_feature = 4
    num_ord_feature = 2
    with open('data/zonesjson.txt') as fp:
        zones = json.loads(fp.readline())
        lvls = json.loads(fp.readline())
    zones = {int(x):zones[x] for x in zones}
    lvls = {int(x):lvls[x] for x in lvls}
    if require_expert:
        fp = open('data/usersjson.txt')
    else:
        fp = open('data/usersjson_sketch.txt')
    users = json.loads(fp.readline())
    users = {int(x):users[x] for x in users if users[x] > 2 * num_frames}
    k = np.array(list(users.keys()), dtype=np.uint64)
    p = np.array(list(users.values()), dtype=np.float32)
    norm = (p - num_frames).sum()
    p /= norm
    p /= p.sum()
    fp.close()
    frame = 0
    net_input = np.zeros((num_frames + skip_frames, num_cat_feature + num_ord_feature, 1), dtype=np.float32)
    reward = 0
    action = 0
    power = 2.3
    while True:
        user = rng.choice(k, p=p)
        start = rng.randint(low=0, high=users[user] - num_frames - skip_frames)
        with open(os.path.join('data/users', '{0}.txt'.format(user))) as fp:
            for idx, line in enumerate(fp):
                if idx >= start and idx < start + num_frames + skip_frames:
                    idx_logstats, user, tt, guild, lvl, race, category, zone, seq = line.strip().split(',')
                    #print(idx, start, start + num_frames + skip_frames - 1)
                    # possible additional features
                    # the time elapse during current zone
                    # the time elapse during current session
                    # the player total time spending
                    net_input[idx - start, :, 0] = np.array([guild, race, category, zone, lvl, idx])
                    if idx == start + num_frames - 1:
                        lvl_in = int(lvl)
                    if idx == start + num_frames + skip_frames - 1:
                        lvl_out = int(lvl)
        reward = np.int64(lvl_out ** power - lvl_in ** power)
        #action = np.argmax(np.bincount(net_input[-skip_frames:, -3, 0].reshape((skip_frames, )).astype(np.uint8)))
        action = net_input[-skip_frames:, -3, 0].reshape((skip_frames, )).astype(np.uint8)
        action_set = lvls[lvl_in]
        if not action[0] in action_set:
            print(action, lvl_in)
            print(zones[action[0]])
        frame += 1
        if frame == num_frames:
            yield (net_input, reward, action, action_set)
            frame = 0
            net_input = np.zeros((num_frames + skip_frames, num_cat_feature + num_ord_feature, 1), dtype=np.float32)
            reward = 0
            action = 0


def batches(batch_size=32):
    g = frames()
    batch = []
    idx = 0
    while True:
        batch.append(g.__next__())
        idx += 1
        if idx == batch_size:
            yield [np.array(x) for x in zip(*batch)]
            idx = 0
            batch = []

def h5dump(path='data/h5', size=100000, batch_size=32):
    s = frames().__next__()
    with open('data/zonesjson.txt') as fp:
        zones = json.loads(fp.readline())
    num_zones = len(zones)
    shape = [x.shape for x in s]
    f = h5py.File('data/episodes', 'w')
    f.create_dataset('context', (size, *shape[0]), 'f')
    f.create_dataset('reward', (size, *shape[1]), 'f')
    f.create_dataset('action', (size, *shape[2]), 'i64')
    f.create_dataset('candidates', (size, num_zones), 'i64')
    
    


def h5():
    
        
class NeuralAgent(object):

    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, exp_pref,
                 replay_start_size, update_frequency, rng):

        self.network = q_network
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_frequency = update_frequency
        self.rng = rng

        self.replay_memory_size = replay_memory_size
        
        self.replay_start_size = replay_start_size

        self.phi_length = self.network.num_frames
        self.frame_length = self.phi_length
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_pref = 'capsule'
        self.exp_dir = 'data/' + self.exp_pref + time_str + \
                       "{}".format(self.network.lr).replace(".", "p") + "_" \
                       + "{}".format(self.network.discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self.num_actions = self.network.num_actions

        self.epsilon = self.epsilon_start
        
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self._open_results_file()
        self._open_learning_file()

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        self.last_action = None

        # Exponential moving average of runtime performance.
        self.steps_sec_ema = 0.

        self.dataset_train = self.frames(-1, require_expert=False)
        self.dataset_test = self.frames(-1, require_expert=True)



    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,mean_q\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                        self.total_reward / float(num_episodes),
                                        holdout_sum)
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action


    def _show_phis(self, phi1, phi2):
        import matplotlib.pyplot as plt
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1

        #TESTING---------------------------
        if self.testing:
            self.episode_reward += reward
            cat_input, ord_input, cat_iutput_prime, ord_input_prime, reward, action, action_set = self.frames.next()
            state = (cat_input, ord_input, cat_iutput_prime, ord_input_prime)
            action_hat = self.network.choose_action(state, epsilon=0)
            accuracy = action_hat == action

        #NOT TESTING---------------------------
        else:
            cat_input, ord_input, cat_iutput_prime, ord_input_prime, reward, action, action_set = self.frames.next()
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)

            state = (cat_input, ord_input, cat_iutput_prime, ord_input_prime)
            action_hat = self.network.choose_action(state, self.epsilon)
            accuracy = action_hat == action

            self.netowrk.train(state, action, action_set, reward)

            if self.step_counter % self.update_frequency == 0:
                loss = self._do_training()
                self.batch_counter += 1
                self.loss_averages.append(loss)

        self.last_action = action
        self.last_img = observation

        return action

    def _choose_action(self, state, epsilon):

        action = self.network.choose_action(state, epsilon)

        return action

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        imgs, actions, rewards, terminals = \
                                self.data_set.random_batch(
                                    self.network.batch_size)
        return self.network.train(imgs, actions, rewards, terminals)


    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:

            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True)

            rho = 0.98
            self.steps_sec_ema *= rho
            self.steps_sec_ema += (1. - rho) * (self.step_counter/total_time)

            logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
                self.step_counter/total_time, self.steps_sec_ema))

            if self.batch_counter > 0:
                self._update_learning_file()
                logging.info("average loss: {:.4f}".format(\
                                np.mean(self.loss_averages)))


    def finish_epoch(self, epoch):
        net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                        '.pkl', 'w')
        cPickle.dump(self.network, net_file, -1)
        net_file.close()

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0

    def finish_testing(self, epoch):
        self.testing = False
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            imgs, _, _, _ = self.data_set.random_batch(holdout_size)
            self.holdout_data = imgs[:, :self.phi_length]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)


if __name__ == "__main__":
    #g = frames()
    g = batches()
    with open('data/zonesjson.txt') as fp:
        zones = json.loads(fp.readline())
    cnt1 = 0
    cnt2 = 1
    for i, x in islice(enumerate(g), 20):
        print(i, x[0].shape)
    #    if x[2] in x[3]:
    #        cnt1 +=1
    #    cnt2 += 1
    #    print('{0}/{1}'.format(cnt1, cnt2))
        #print(x[0].shape, x[1], x[2], x[3])
