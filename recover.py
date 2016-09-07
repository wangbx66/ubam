# -*- coding: utf-8 -*-

import logging
import os
import pickle
import itertools

import lasagne
import numpy as np
import theano
import theano.tensor as TT
from scipy.optimize import linprog

from architecture import build_wowah_network
from architecture import major
from agent import frames
from agent import hdf

def q_tune(n_r, skip_frames, hdf_file='data/episodes.hdf'):
    l_outs = list(range(n_r))
    q_vals = list(range(n_r))
    q_funcs = list(range(n_r))
    for i in range(n_r):
        states = TT.tensor4('states')
        pp = pickle.load(open('data/Q{0}.pkl'.format(i), 'rb'))
        l_outs[i] = build_wowah_network()
        lasagne.layers.set_all_param_values(l_outs[i], pp)
        q_vals[i] = lasagne.layers.get_output(l_outs[i], states)
        q_funcs[i] = theano.function(inputs=[states], outputs=q_vals[i])
        num_batches = 100
        data = hdf(path=hdf_file, batch_size=1000, num_batch=num_batches)
        print("Q functions #{0} loaded. Tuning now ...".format(i))
        mean = 0
        for context, rewards, action, candidates in data:
            q_val_eval = q_funcs[i](context[:, :-skip_frames])
            mean += q_val_eval.mean() / num_batches
        print("Q functions #{0} divided by {1}".format(i, mean))
        q_funcs[i] = theano.function(inputs=[states], outputs=q_vals[i] / mean)
    return q_funcs

def cons_gen(trajsjson='data/trajsjson.txt'):
    trajs = []

n_r = 5
skip_frames = 4
q_funcs = q_tune(n_r, skip_frames)

batch_size = 1
num_zones = 165
C = 1
n_trajs = 1000
q_val_eval = list(range(n_r))
d_q = np.zeros((num_zones, n_r))
data = hdf(path='data/episodes.hdf', batch_size=batch_size, num_batch=n_trajs)
A = []
for context, rewards, action, candidates in data:
    action = major(action, 4, 1)[0]
    candidates = candidates.flatten()
    for i in range(n_r):
        q_val_eval = q_funcs[i](context[:, :-skip_frames]).flatten()
        d_q[:, i] = q_val_eval - q_val_eval[action]
    for i in range(len(candidates)):
        if candidates[i] == 1 and not i == action:
            A.append(d_q[i, :])
A = np.array(A)
A = np.concatenate((A, -np.eye(A.shape[0])), axis=1)
b = np.zeros(A.shape[0])
c = np.concatenate((np.zeros(n_r), C * np.ones(A.shape[0])), axis=0)
reg = np.concatenate((np.ones((1, n_r)), np.zeros((1, A.shape[0]))), axis=1)
norm = np.ones(1)
sol = linprog(c, A, b, reg, norm)
