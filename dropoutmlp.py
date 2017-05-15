 #!/usr/bin/python
#coding: utf-8
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time

import numpy

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
import theano.tensor as T
import pdb


from logistic_sgd import LogisticRegression, load_data
from load_data import load_mnist

def load_mnist_data(dataset):
    if dataset.endswith('.npz'):
        print 'load npz'
        datasets = load_mnist(dataset)
    elif datasets.endswith('.gz'):
        print 'load gz'
        datasets = load_data(dataset)
    else:
        print 'can not load dataset:', dataset 
        assert False
    return datasets

def AddLayer(rng, input, n_in, n_out, W=None, b=None, args = None, type='hidden'):
    assert type in ['hidden', 'dropout', 'adaptive_dropout', 'dropconnect']
    if type == 'hidden':
        return HiddenLayer(rng, input, n_in, n_out, W=W, b=b, args=args)
    elif type == 'dropout':
        return DropoutHiddenLayer(rng, input, n_in, n_out, W=W, b=b, args=args)
    elif type == 'adaptive_dropout':
        return AdaptiveDropoutHiddenLayer(rng, input, n_in, n_out, W=W, b=b, args=args)
    else:
        return DropConnectHiddenLayer(rng, input, n_in, n_out, W=W, b=b, args=args)

class HiddenLayer(object):             #这个类只构造了一层隐层。
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, args = None):
        self.input = input
        activation = get_act_function(args.act_type)
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)      #初始化权值 
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            if rbm == True:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)      #把权值共享了
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        activation = get_act_function(args.act_type)
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

def get_act_function(func_name = 'sigmoid'):
    assert func_name in ['sigmoid', 'relu', 'tanh', None]
    print ('activation: ' + func_name)
    if func_name == 'sigmoid':
        return T.nnet.sigmoid
    elif func_name == 'relu':
        return lambda x: T.maximum(0.0, x)
    elif func_name == 'tanh':
        return T.tanh
    else:
        return None

def get_mask(shape, rng, args):
    """ for code briefness, put all parameters into args
        args = {p or (mu, sigma) or (a, b)}
        for example: args.p = 0.5
    """
    assert args.drop_type in ['bernoulli', 'gaussian', 'uniform']
    srng = RandomStreams(rng.randint(99999))
    drop_type = args.drop_type
    print('dropout_type: ' + drop_type)
    if drop_type == 'bernoulli':
        return srng.binomial(n = 1, p = 1 - args.p, size = shape)
    elif drop_type == 'gaussian':
        mask = srng.normal(avg  = args.mu, std = args.sigma, size = shape)
        if not args.noclip:
            print('clip')
            mask = T.clip(mask, 0., 1.)
        else:
            print('noclip')
        return  mask
    elif drop_type == 'uniform':
        return srng.uniform(low = args.a, high = args.b, size = shape)
        
def  _drop_out_from_layer(rng, layer, args):
    mask = get_mask(layer.shape, rng, args)
    return layer * T.cast(mask,theano.config.floatX)

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 args = None, rbm = False):
        """ required args:
            args.activation: activation type name
            args.drop_type: dropout type name
            args.{p or (mu, sigma) or (a, b)}
        """
        super(DropoutHiddenLayer, self).__init__(
              rng=rng, input=input,n_in=n_in,n_out=n_out,W=W,b=b,
              args = args, rbm = rbm)#在新式函数中调用超类
        self.output = _drop_out_from_layer(rng, self.output, args)  

class DropConnectHiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 args=None):
        self.input = input
        activation = get_act_function(args.act_type)
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)      #初始化权值 

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)      #把权值共享了

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        srng = RandomStreams(rng.randint(99999))
        mask = srng.binomial(n=1, p=1-args.p, size= self.W.shape)
        mask  =  T.cast(mask, theano.config.floatX)
        dropped_W = self.W * mask
        lin_output = T.dot(input, dropped_W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

class AdaptiveDropoutHiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 args=None):
        self.input = input
        activation = get_act_function(args.act_type)
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)      #初始化权值 

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)      #把权值共享了

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        prob = T.dot(input, self.W) + self.b

        prob = T.nnet.sigmoid(args.alpha * prob + args.beta)

        srng = RandomStreams(rng.randint(99999))
        mask = srng.binomial(size = prob.shape, p = prob)
        mask  =  T.cast(mask, theano.config.floatX)

        lin_output = T.dot(input, self.W) + self.b
        act_output = (lin_output if activation is None
                       else activation(lin_output))

        self.output = mask * act_output

        # parameters of the model
        self.params = [self.W, self.b]
