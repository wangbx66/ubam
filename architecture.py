# -*- coding: utf-8 -*-

from collections import OrderedDict
from itertools import islice
import logging
import inspect
import os
import sys
import pickle

import lasagne
import numpy as np
import theano
import theano.tensor as TT

from agent import hdf

def logging_config(name=None, file_level=logging.DEBUG, console_level=logging.DEBUG):
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to", logpath)
    logging.root.setLevel(file_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(file_level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    logconsole = logging.StreamHandler()
    logconsole.setLevel(console_level)
    logconsole.setFormatter(formatter)
    logging.root.addHandler(logconsole)

def rmsprop(loss_or_grads, params, learning_rate, rho, epsilon):
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)

        acc_grad = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        acc_grad_new = rho * acc_grad + (1 - rho) * grad
        acc_rms = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        acc_rms_new = rho * acc_rms + (1 - rho) * grad ** 2

        updates[acc_grad] = acc_grad_new
        updates[acc_rms] = acc_rms_new
        updates[param] = (param - learning_rate * (grad / TT.sqrt(acc_rms_new - acc_grad_new **2 + epsilon)))

    return updates, grads

class T_T(lasagne.layers.Layer):
    '''
    卧槽我就乘个矩阵还要自己写 这TM是在逗我 (｡ŏ﹏ŏ)
    '''
    def __init__(self, incoming, num_units, W, b, nonlinearity, **kwargs):
        super(T_T, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[2:]))

        self.W = self.add_param(W, (num_inputs, num_units, 1), name="WT_T")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units, 1)

    def get_output_for(self, input_tensor, **kwargs):
        if input_tensor.ndim > 3:
            input_tensor = input_tensor.flatten(3)

        activation = TT.tensordot(input_tensor, self.W, axes=(2, 0))
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 'x', 0, 'x')
        return self.nonlinearity(activation)

class QwQ(lasagne.layers.Layer):
    '''
    改个type也不行.. Excuse me???? (｡ŏ﹏ŏ)
    '''
    def __init__(self, incoming, dtype, **kwargs):
        super(QwQ, self).__init__(incoming, **kwargs)
        self.dtype = dtype

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input_tensor, **kwargs):
        return input_tensor.astype(self.dtype)

def build_wowah_network(num_frames=10, input_width=8, cat_length=5, cat_size=[513,5,10,165,9], output_dim=165):

    # int(guild), int(race), int(category), int(zone), int(zonetype), norm_lvl, norm_num_zones, norm_zone_stay

    input_height = 1

    context = TT.tensor4(name='input')
    l_in = lasagne.layers.InputLayer(
        shape=(None, num_frames, input_width, input_height),
        input_var=context,
    )

    cat_h = []
    for idx in range(cat_length):
        cat_slice = lasagne.layers.SliceLayer(
            l_in,
            indices=idx,
            axis=2,
        )

        cat_int = QwQ(
            cat_slice,
            dtype='int16',
        )

        cat_embed = lasagne.layers.EmbeddingLayer(
            cat_int,
            input_size=cat_size[idx],
            output_size=10,
            W=lasagne.init.Normal(.01),
        )

        cat_reshape = lasagne.layers.DimshuffleLayer(
            cat_embed,
            # pattern=(0, 1, 3, 'x'),
            # we omit the trailing 'x', only for wowah dataset s.t. height=1
            pattern=(0, 1, 3, 2)
        )

        cat_h.append(cat_reshape)

    ord_slice = lasagne.layers.SliceLayer(
            l_in,
            indices=slice(cat_length, None),
            axis=2,
        )

    ord_h = T_T(
        ord_slice,
        num_units=30,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_hidden1 = lasagne.layers.ConcatLayer(
        cat_h + [ord_h, ],
        axis=2,
    )

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=120,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden2,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )
    
    sample = np.zeros((32, num_frames, input_width,input_height), dtype=np.float32)
    sample[:,:,:cat_length,:] = sample[:,:,:cat_length,:].astype(np.uint32)

    context.tag.test_value = sample
    
    return l_out#, locals()

def major(actions, skip_frames, batch_size):
    z = []
    for i in range(batch_size):
        s = {}
        for j in range(skip_frames):
            a = actions[i, j]
            if a in s:
                s[a] += 1
            else:
                s[a] = 1
        m = max(list(s.values()))
        if m > 1:
            for j in range(skip_frames):
                a = actions[i, -j-1]
                if s[a] == m:
                    z.append(a)
                    s = {}
                    break
        else:
            z.append(actions[i, -1])
            s = {}
    return np.array(z, dtype=np.uint32)

if __name__ == '__main__':
    if not os.path.exists('data/networks'):
        os.mkdir('data/networks')
    
    reward_idx = int(sys.argv[1])
    num_batch = int(sys.argv[2])
    lr = float(sys.argv[3]) # 0.00045
    
    logging_config()
    
    theano.config.optimizer = 'fast_run'
    theano.config.exception_verbosity = 'low'
    theano.config.compute_test_value = 'off'
    #theano.config.optimizer = 'fast_compile'
    #theano.config.exception_verbosity = 'high'
    
    l_out = build_wowah_network()

    batch_size = 32
    #num_batch = 3000
    test_batch = 100
    num_frames = 10
    skip_frames = 4
    input_width = 8
    input_height = 1
    discount = 0.99
    clip_delta = 1.0
    #lr = 0.00005
    rho = 0.95
    rms_epsilon = 0.01
    num_actions = 165
    #reward_idx = 0

    states = TT.tensor4('states')
    next_states = TT.tensor4('next_states')
    rewards = TT.col('rewards')
    actions = TT.col('actions', dtype='uint32')

    context_shared = theano.shared(
        np.zeros((batch_size, num_frames + skip_frames, input_width, input_height),
                 dtype=theano.config.floatX))
    rewards_shared = theano.shared(
        np.zeros((batch_size, 1), dtype=theano.config.floatX),
        broadcastable=(False, True))
    actions_shared = theano.shared(
        np.zeros((batch_size, 1), dtype='uint32'),
        broadcastable=(False, True))
    state_shared = theano.shared(
        np.zeros((num_frames, input_height, input_width),
                 dtype=theano.config.floatX))
    
    q_vals = lasagne.layers.get_output(l_out, states)
    next_q_vals = lasagne.layers.get_output(l_out, next_states)
    #next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

    actionmask = TT.eq(TT.arange(num_actions).reshape((1, -1)), actions.reshape((-1, 1))).astype(theano.config.floatX)
    target = (rewards + discount * TT.max(next_q_vals, axis=1, keepdims=True))
    output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
    diff = target - output
    quadratic_part = TT.minimum(abs(diff), clip_delta)
    linear_part = abs(diff) - quadratic_part
    loss_batch = 0.5 * quadratic_part ** 2 + clip_delta * linear_part
    loss = TT.sum(loss_batch)
    
    train_subs = {
        states: context_shared[:, :-skip_frames],
        next_states: context_shared[:, skip_frames:],
        rewards: rewards_shared,
        actions: actions_shared,
    }
    params = lasagne.layers.helper.get_all_params(l_out)
    updates, grads = rmsprop(loss, params, lr, rho, rms_epsilon)
    speed = sum([grad.norm(L=2) for grad in grads]) / sum([grad.shape.prod() for grad in grads])

    train = theano.function(inputs=[], outputs=[loss, speed], updates=updates, givens=train_subs)
    # we evaluate it using value programming
    # actions_hat = TT.argmax(q_vals, axis=1)
    q_func = theano.function(inputs=[states], outputs=q_vals)

    for epoch in range(20000):
        total_loss = 0
        total_speed = 0    
        hits = 0
        stays = 0
        predict_stays = 0
        for idx, (context, reward, action, candidates) in enumerate(islice(hdf(batch_size=batch_size, num_batch=num_batch), num_batch)):
            action_star = major(action, skip_frames, batch_size)
            context_shared.set_value(context)
            rewards_shared.set_value(reward[:,reward_idx].reshape(batch_size, 1))
            actions_shared.set_value(action_star.reshape(batch_size, 1))

            train_results = train()
            loss, speed = train_results
            #speed = train_results[1:]
            #speed = [float(x) for x in speed]
            total_loss += loss
            total_speed += speed

            if idx >= num_batch - test_batch:
                q_hat = q_func(context[:, :-skip_frames])
                last = context[:,-(skip_frames+1),3,:].flatten()
                actions_hat = np.argmax(q_hat * candidates, axis=1)
                hits += (actions_hat == action_star).sum()
                stays += (last == action_star).sum()
                predict_stays += (last == actions_hat).sum()
                #print(actions_hat, action_star)
                #print(hits)
        accuracy = hits / (batch_size * test_batch)
        stick = stays / (batch_size * test_batch)
        predict_stick = predict_stays / (batch_size * test_batch)
        logging.info('epoch #{3}: loss = {0}, speed = {1}, accuracy = {2}'.format(total_loss, total_speed, accuracy, epoch+1))
        print('stay rate = {0}, unary rate = {1}'.format(stick, predict_stick))
        logging.info(str(np.argmax(q_hat, axis=1)))
        network = lasagne.layers.get_all_param_values(l_out)
        netfile = open('data/networks/Q-{0}-{1}-{2}.pkl'.format(reward_idx, epoch+1, accuracy), 'wb')
        pickle.dump(network, netfile)
            
