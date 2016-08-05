import lasagne
import numpy as np
import theano
import theano.tensor as TT

from updates import deepmind_rmsprop
from architecture import build_wowah_network

class DeepQLearner:

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0):

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng

        lasagne.random.set_rng(self.rng)

        self.update_counter = 0

        self.l_out = self.build_network(network_type, input_width, input_height,
                                        num_actions, num_frames, batch_size)
        if self.freeze_interval > 0:
            self.next_l_out = self.build_network(network_type, input_width,
                                                 input_height, num_actions,
                                                 num_frames, batch_size)
            self.reset_q_hat()

        states = TT.tensor4('states')
        next_states = TT.tensor4('next_states')
        rewards = TT.col('rewards')
        actions = TT.icol('actions')
        terminals = TT.icol('terminals')

        # Shared variables for training from a minibatch of replayed
        # state transitions, each consisting of num_frames + 1 (due to
        # overlap) images, along with the chosen action and resulting
        # reward and terminal status.
        self.imgs_shared = theano.shared(
            np.zeros((batch_size, num_frames + 1, input_height, input_width),
                     dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        # Shared variable for a single state, to calculate q_vals.
        self.state_shared = theano.shared(
            np.zeros((num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)
        
        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                                                    next_states / input_scale)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                                                    next_states / input_scale)
            next_q_vals = theano.gradienTT.disconnected_grad(next_q_vals)

        terminalsX = terminals.astype(theano.config.floatX)
        actionmask = TT.eq(TT.arange(num_actions).reshape((1, -1)),
                          actions.reshape((-1, 1))).astype(theano.config.floatX)

        target = (rewards +
                  (TT.ones_like(terminalsX) - terminalsX) *
                  self.discount * TT.max(next_q_vals, axis=1, keepdims=True))
        output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
        diff = target - output

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring \partial loss /\partial q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = TT.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = TT.sum(loss)
        elif batch_accumulator == 'mean':
            loss = TT.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(self.l_out)
        train_givens = {
            states: self.imgs_shared[:, :-frame_skip],
            next_states: self.imgs_shared[:, frame_skip:],
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([], [loss], updates=updates,
                                      givens=train_givens)
        q_givens = {
            states: self.state_shared.reshape((1,
                                               self.num_frames,
                                               self.input_height,
                                               self.input_width))
        }
        self._q_vals = theano.function([], q_vals[0], givens=q_givens)

    def build_network(self, input_width, input_height,
                      output_dim, num_frames, cat_length, cat_size):

        return build_wowah_network(num_frames, input_width, cat_length, cat_size, output_dim)


    def train(self, imgs, actions, rewards, terminals):
        """
        Train one batch.

        Arguments:

        imgs - b x (num_frame + frame_skip) x width x 1 numpy array
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """

        self.imgs_shared.set_value(imgs)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state):
        self.state_shared.set_value(state)
        return self._q_vals()

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)


def main():
    net = DeepQLearner(84, 84, 16, 4, .99, .00025, .95, .95, 10000,
                       32, 'nature_cuda')

def flowtest():
    from agent import frames, batches
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # \rho
    RMS_EPSILON = .01
    MOMENTUM = 0
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 4
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "nature_dnn"
    FREEZE_INTERVAL = 10000
    REPLAY_START_SIZE = 50000
    RESIZE_METHOD = 'scale'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 30
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False
    s = DeepQLearner(input_width=6, 
                input_height=1, 
                num_actions=165,
                num_frames=10, 
                discount=0.99, 
                learning_rate=0.00025, 
                rho=0.95,
                rms_epsilon=0.01, 
                momentum=0, 
                clip_delta=1.0, 
                freeze_interval=100,
                batch_size=32, 
                network_type='notdeepatallqnetwork', 
                update_rule,
                batch_accumulator='sum', 
                rng = np.random.RandomState(123456))

if __name__ == '__main__':
    flowtest()
    
