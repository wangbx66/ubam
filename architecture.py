# -*- coding: utf-8 -*-

import lasagne
import numpy as np
import theano
import theano.tensor as TT
from lasagne.layers import get_output_shape as shape

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

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input_tensor, **kwargs):
        if input_tensor.ndim > 3:
            input_tensor = input_tensor.flatten(3)

        #activation = TT.tensordot(input_tensor, self.W, axes=(2, 1))
        activation = TT.dot(input_tensor, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class QwQ(lasagne.layers.Layer):
    '''
    改个type也不行.. Excuse me???? (｡ŏ﹏ŏ)
    '''
    def __init__(self, incoming, dtype, **kwargs):
        super(T_T, self).__init__(incoming, **kwargs)
        self.dtype = dtype

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input_tensor, **kwargs):
        return input_tensor.astype(dtype)

def build_wowah_network(num_frames=50, input_width=6, cat_length=4, cat_size=[10,11,12,13], output_dim=160):

    input_height = 1

    sample = np.zeros((32,50,6,1), dtype=np.float32)
    sample[:,:,:4,:] = sample[:,:,:4,:].astype(np.uint8)

    l_in = lasagne.layers.InputLayer(
                shape=(None, num_frames, input_width, input_height)
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
            pattern=(0, 1, 3, 'x'),
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

    return l_out
