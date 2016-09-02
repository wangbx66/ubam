# -*- coding: utf-8 -*-

from collections import OrderedDict
import logging
import inspect
import os
import pickle

import lasagne
import numpy as np
import theano
import theano.tensor as TT

from architecture import build_wowah_network

l_outs = []
for i in range(5):
    l_out = build_wowah_network()
    pp = pickle.load('data/Q{0}.pkl'.format(i))
    lasagne.layers.set_all_param_values(l_out, pp)
    l_outs.append(l_out)



