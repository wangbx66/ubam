# -*- coding: utf-8 -*-

from collections import OrderedDict

import lasagne
import numpy as np
import theano
import theano.tensor as TT

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

    return updates

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

def build_wowah_network(num_frames=10, input_width=6, cat_length=4, cat_size=[512,5,10,165], output_dim=165):

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
            dtype='uint8',
        )

        cat_embed = lasagne.layers.EmbeddingLayer(
            cat_int,
            input_size=cat_size[idx],
            output_size=25,
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
        num_units=50,
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
        num_units=500,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden2,
        num_units=output_dim,
        nonlinearity=None,
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )
    
    sample = np.zeros((32,10,6,1), dtype=np.float32)
    sample[:,:,:4,:] = sample[:,:,:4,:].astype(np.uint32)

    context.tag.test_value = sample
    
    return l_out, locals()

def major(actions):
    z = []
    for i in range(batch_size):
        s = {}
        for j in range(skip_frames):
            a = action[i, j]
            if a in s:
                s[a] += 1
            else:
                s[a] = 1
        m = max(list(s.values()))
        if m > 1:
            for j in range(skip_frames):
                a = action[i, -j-1]
                if s[a] == m:
                    z.append(a)
                    s = {}
                    break
        else:
            z.append(action[i, -1])
            s = {}
    return np.array(z, dtype=np.uint32)

if __name__ == '__main__':
    from agent import hdf
    from itertools import islice
    
    #theano.config.optimizer = 'fast_compile'
    #theano.config.exception_verbosity = 'high'
    theano.config.optimizer = 'fast_run'
    theano.config.exception_verbosity = 'low'
    theano.config.compute_test_value = 'off'
    
    l_out, netcat = build_wowah_network()

    batch_size = 32
    num_frames = 10
    skip_frames = 4
    input_width = 6
    input_height = 1
    discount = 0.99
    clip_delta = 1.0
    lr = 0.00025
    rho = 0.95
    rms_epsilon = 0.01
    num_actions = 165

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
    next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

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
    updates = rmsprop(loss, params, lr, rho, rms_epsilon)

    train = theano.function(inputs=[], outputs=[loss], updates=updates, givens=train_subs)
    # we evaluate it using value programming
    # actions_hat = TT.argmax(q_vals, axis=1)
    Q = theano.function(inputs=[states], outputs=q_vals)

    accuracy = 0
    num_batch = 3000
    for idx, (context, reward, action, candidates) in enumerate(islice(hdf(batch_size=batch_size, num_batch=num_batch), num_batch)):
        if idx % 1000 == 0:
            print(idx)
        action_star = major(action)
        context_shared.set_value(context)
        rewards_shared.set_value(reward.reshape(batch_size, 1))
        actions_shared.set_value(action_star.reshape(batch_size, 1))

        loss = train()
        print('loss = {0}'.format(loss))
        
        #q_hat = Q(context[:, :-skip_frames])
        #actions_hat = np.argmax(q_hat * candidates, axis=1)
        #accuracy += (actions_hat == action_star).sum()
    #print(accuracy / (batch_size * num_batch / 2))
        
        
