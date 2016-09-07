# -*- coding: utf-8 -*-

import logging
import os
import pickle
import itertools

import lasagne
import numpy as np
import theano
import theano.tensor as TT

from architecture import build_wowah_network
from architecture import major
from agent import frames
from agent import batches

n_r = 5
l_outs = list(range(n_r))
q_vals = list(range(n_r))
q_func = list(range(n_r))
for i in range(n_r):
    states = TT.tensor4('states')
    pp = pickle.load(open('data/Q{0}.pkl'.format(i), 'rb'))
    l_outs[i] = build_wowah_network()
    lasagne.layers.set_all_param_values(l_outs[i], pp)
    q_vals[i] = lasagne.layers.get_output(l_outs[i], states)
    q_func[i] = theano.function(inputs=[states], outputs=[q_vals[i]])

C = 1
n_cons = 1500
trajsjson = 'data/trajsjson.txt'
trajs = []
q_val_eval = list(range(n_r))
d_q = np.zeros((n_r, ))
s = batches(trajsjson=trajsjson, batch_size=1)
A_ub = []
for x in itertools.islice(s, n_cons):
    context, rewards, action, candidates = x
    action = major(action, 4, 1)
    for i in range(n_r):
        q_val_eval = q_func[i](context)
        d_q[i] = q_val_eval - q_val_eval[action]
        action_value = q_val_eval[i]
    for i in range(len(candidates)):
        if candidates[i] == 1:
            A_ub.append(d_q)
    
