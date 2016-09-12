# -*- coding: utf-8 -*-

import logging
import os
import pickle
import itertools
import collections

import lasagne
import numpy as np
import theano
import theano.tensor as TT
import pulp
from scipy.optimize import linprog

from architecture import build_wowah_network
from architecture import major
from agent import frames
from agent import hdf
from agent import hdf_dump

def q_tune(n_r, skip_frames, hdf_file='data/episodes.hdf'):
    l_outs = list(range(n_r))
    q_vals = list(range(n_r))
    q_funcs = list(range(n_r))
    A = [20, 5, 5, 5, 5]
    for i in range(n_r):
        states = TT.tensor4('states')
        pp = pickle.load(open('data/Q{0}.pkl'.format(i), 'rb'))
        l_outs[i] = build_wowah_network()
        lasagne.layers.set_all_param_values(l_outs[i], pp)
        q_vals[i] = lasagne.layers.get_output(l_outs[i], states)
        q_funcs[i] = theano.function(inputs=[states], outputs=q_vals[i])
        num_batches = 100
        data = hdf(path=hdf_file, batch_size=1000, num_batch=num_batches)
        #print("Q functions #{0} loaded. Tuning now ...".format(i))
        mean = 0
        for context, rewards, action, candidates in data:
            q_val_eval = q_funcs[i](context[:, :-skip_frames])
            mean += q_val_eval.mean() / num_batches
        #print("Q functions #{0} divided by {1}".format(i, mean))
        q_funcs[i] = theano.function(inputs=[states], outputs=100*q_vals[i]*A[i] / mean)
    return q_funcs

def pos(x):
    return(x * (x > 0))

def neg(x):
    return(x * (x < 0))

def universal():
    return 'data/trajsjson.txt', 'trajsjson', None

def timeinterval(ll, uu):
    def flt(idx_logstats, user, tt, guild, lvl, race, category, zone, seq, zonetype, num_zones, zone_stay, r1, r2, r3, r4, r5):
        return ll <= tt <= uu
    return 'data/trajsjson.txt', 'time-{0}-{1}'.format(ll, uu), flt

def certainjson(jsonfile):
    return jsonfile, 'json-{0}'.format(jsonfile), None

def cons_gen(trajsjson, name, flt):
    n_r = 5
    skip_frames = 4
    q_funcs = q_tune(n_r, skip_frames)
    batch_size = 1
    num_zones = 165
    C = 1
    n_trajs = 1500
    q_val_eval = list(range(n_r))
    d_q = np.zeros((num_zones, n_r))
    A = []
    
    #name, flt = universal()
    if not os.path.exists('data/episodes'):
        os.mkdir('data/episodes')
    path = 'data/episodes/{0}_{1}.hdf'.format(name, n_trajs)
    if not os.path.exists(path):
        hdf_dump(trajsjson=trajsjson, path=path, size=n_trajs, flt=flt)
    data = hdf(path=path, batch_size=batch_size, num_batch=n_trajs)
    for context, rewards, action, candidates in data:
        action = major(action, 4, 1)[0]
        candidates = candidates.flatten()
        for i in range(n_r):
            q_val_eval = q_funcs[i](context[:, :-skip_frames]).flatten()
            d_q[:, i] = q_val_eval - q_val_eval[action]
        for i in range(num_zones):
            if candidates[i] == 1 and not i == action:
                A.append(d_q[i, :].copy())
    A = np.array(A)
    return A

def scipysol(A, b, c):
    n_cons = A.shape[0]
    A = np.concatenate((A, -np.eye(n_cons)), axis=1)
    A = np.concatenate((A, np.concatenate((-np.ones((1, n_r)), np.zeros((1, A.shape[0]))), axis=1)), axis=0)
    b = np.concatenate((np.zeros(n_cons), -np.ones(1)))
    c = np.concatenate((np.zeros(n_r), C * np.ones(n_cons)), axis=0)
    sol = linprog(c, A_ub=A, b_ub=b, options={'maxiter': 10000})
    return sol

def pulpsol(A):
    variables = list(range(A.shape[0]+5))
    x = pulp.LpVariable.dicts('vars', variables, lowBound=0)
    lp_model = pulp.LpProblem('recover', pulp.LpMinimize)
    lp_model += sum(x[var] for var in variables if var >=5)
    for i in range(A.shape[0]):
        lp_model += sum(A[i,var] * x[var] for var in range(5)) <= x[i+5]
    lp_model += sum([x[var] for var in range(5)]) >= 1
    lp_model.solve()
    return lp_model, x

if __name__ == '__main__':
    #for ll in range(120000, 200000, 4320):
    ll=149000
    A = cons_gen(*timeinterval(ll, ll+4320))
    s = pos(A)
    t = neg(A)
    phi = s.mean(axis=0)
    print(phi*1.01/phi.sum())
    per = [int(0.5+x*100/phi.sum()) for x in phi]
    print(per)
    lp_model, x = pulpsol(A)

