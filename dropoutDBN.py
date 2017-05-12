 #!/usr/bin/python
#coding: utf-8
import cPickle  #解释见：http://flyfeeling.blogbus.com/logs/64147735.html
import gzip
import os
import sys
import time
import pdb

import numpy
import numpy.distutils
import numpy.distutils.__config__
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from logistic_sgd import LogisticRegression, load_data
from dropoutmlp import HiddenLayer,DropoutHiddenLayer,_drop_out_from_layer, \
    get_act_function, load_mnist_data
from rbm import RBM
from load_data import load_mnist
import theano.printing
import argparse
import copy
import logging, random
from sys import stderr

class DBN(object):
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[400, 400], n_outs=10, drop_input = True,
                 args = None, rbm = False):
        """ required fields in args
            args.drop_type
            args.{p or (mu, sigma) or (a, b)}
            args.drop_input
            args.rbm
            args.act_type
        """
        self.layers = []   
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.dropout_layers = []

        assert self.n_layers > 0     

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))    
        self.x = T.matrix('x')  # the data is presented as rasterized images    
        self.y = T.ivector('y')  # the labels are presented as 1D vector # of [int] labels


        for i in xrange(self.n_layers):       #这里进行一个循环，构造DBN # construct the sigmoidal layer 
            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:       # 如果是第一层
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]  

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer

            if i == 0:
                print >> stderr, 'build input layer:'
                layer_input = self.x
                if drop_input:
                    ### drop input by 0.2 without change the following dropout
                    ### settings
                    args0 = copy.copy(args)
                    args0.drop_type = 'bernoulli'
                    args0.p = 0.2
                    dropout_layer_input = _drop_out_from_layer(numpy_rng, self.x, args0)
                else:
                    dropout_layer_input = self.x
            else:
                layer_input = self.layers[-1].output      
                dropout_layer_input = self.dropout_layers[-1].output

            print >> stderr, "build %dth dropout layer, input size: %d, output size: %d" \
                % (i, input_size, hidden_layers_sizes[i])

            next_dropout_layer=DropoutHiddenLayer(rng=numpy_rng,
                                             input=dropout_layer_input,
                                             n_in=input_size,
                                             n_out=hidden_layers_sizes[i], 
                                             args = args)

            self.dropout_layers.append(next_dropout_layer)
            print >> stderr, "build %dth normal layer, input size: %d, output size: %d" \
                % (i, input_size, hidden_layers_sizes[i])
            if i==0:
                next_layer = HiddenLayer(rng=numpy_rng,
                                input=layer_input,
                                n_in=input_size,
                                n_out=hidden_layers_sizes[i],
                                W=next_dropout_layer.W * 0.8 if drop_input else 1.,
                                b=next_dropout_layer.b,
                                args = args,
                                rbm = rbm)            
            else:
                next_layer = HiddenLayer(rng=numpy_rng,
                                input=layer_input,
                                n_in=input_size,
                                n_out=hidden_layers_sizes[i],
                                W=next_dropout_layer.W * 0.5,
                                b=next_dropout_layer.b,
                                args = args,
                                rbm = rbm)            

            # add the layer to our list of layers
            self.layers.append(next_layer)       
            
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=next_dropout_layer.W,
                            hbias=next_dropout_layer.b)

            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.dropout_logLayer = LogisticRegression(
            input=self.dropout_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.dropout_layers.append(self.dropout_logLayer)

        self.logLayer = LogisticRegression(
            input=self.layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs,
            W=self.dropout_logLayer.W * 0.5,
            b=self.dropout_logLayer.b)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.dropout_finetune_cost = self.dropout_logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
        # get all the params together
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]

    def build_finetune_functions(self, datasets, batch_size, learning_rate,
            dropout = True, constraint = 15.):
        print "constraint:", constraint
        squared_filter_length_limit =  constraint
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        print >> stderr, "train batches:", n_train_batches
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        print >> stderr, "valid batches:", n_valid_batches
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size
        print >> stderr, 'test_batches:', n_test_batches

        index = T.lscalar('index')  # index to a [mini]batch
        mom=T.scalar('mom')
        learning_rate=T.scalar('lr')

        # Build the expresson for the cost function.
        cost = self.finetune_cost
        dropout_cost = self.dropout_finetune_cost

        # Compute gradients of the model wrt parameters
        gparams = []
        for param in self.params:
        # Use the right cost function here to train with or without dropout.
            gparam = T.grad(dropout_cost if dropout else cost, param)
            gparams.append(gparam)

        # ... and allocate mmeory for momentum'd versions of the gradient
        gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                dtype=theano.config.floatX))
            gparams_mom.append(gparam_mom)

        # Update the step direction using momentum
        updates = {}
        for gparam_mom, gparam in zip(gparams_mom, gparams):
            updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam

        # ... and take a step along that direction
        for param, gparam_mom in zip(self.params, gparams_mom):
            stepped_param = param - learning_rate * gparam_mom

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices.  This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
            if param.get_value(borrow=True).ndim == 2:
                squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
                scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param


        # Compile theano function for training.  This returns the training cost and
        # updates the model parameters.
        print stderr, 'dropout:', dropout
        output = dropout_cost if dropout else cost

        # Compile theano function for training.  This returns the training cost and
        # updates the model parameters.

        train_fn = theano.function(inputs=[index,mom,learning_rate],
              outputs = self.errors,
              updates=updates,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})
        #theano.printing.pydotprint(train_fn, outfile="train_file.png",
        #        var_with_name_simple=True)
        # Compile theano function for testing.
        test_score_i = theano.function([index], self.errors,
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y: test_set_y[index * batch_size:
                                            (index + 1) * batch_size]})
        #theano.printing.pydotprint(test_model, outfile="test_file.png",
        #        var_with_name_simple=True)
        # Compile theano function for validating.
        valid_score_i = theano.function([index], self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})
        #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
        #        var_with_name_simple=True)
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score 

    def save_params():
        pass

    def load_params(path):
        pass

def test_DBN(finetune_lr=1.0, pretraining_epochs=100,
                 pretrain_lr=0.01, k = 1, 
                 training_epochs = 3000,
                 dataset='./data/mnist.pkl.gz', 
                 seed=123,
                 batch_size=100, 
                 learning_rate_decay=0.998,
                 dropout = False,
                 layers=[800,800],
                 results_file_name = 'result/results_dropout',
                 dropout_args = None):
    layers_str = str(layers).strip()
    result_file_path = results_file_name + layers_str
    result_file = open(result_file_path,'w')
    configure = 'configer: '+str(training_epochs)+' epoch for finetune, layers '+str(layers)+\
               ',random seed '+str(seed)+', batch_size_'+str(batch_size)+'.\n'
    result_file.writelines(configure)

    datasets = load_mnist_data(dataset)
    (train_set_x, train_set_y) = datasets[0]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(seed)    #123只是一个随机种子
    if seed == None:
        numpy_rng = numpy.random.RandomState(random.randint(1, 2**30))
    print stderr, '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
                  hidden_layers_sizes=layers,
                  n_outs=10,
                  args = dropout_args,
                  drop_input = dropout)

    #########################
    # PRETRAINING THE MODEL #
    #########################

    ######################## # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print >> stderr,  '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
                datasets, batch_size, finetune_lr, dropout=dropout)

    print  >> stderr, '... finetunning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    learning_rate=finetune_lr

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        learning_rate=learning_rate*learning_rate_decay
        mom = 0.5*(1. - epoch/500.) + 0.99*(epoch/500.) if 500>epoch else 0.99
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index,mom,learning_rate)
            #iter = (epoch - 1) * n_train_batches + minibatch_index
            #if (iter + 1) % validation_frequency == 0:#每次容忍的迭代的epoch为当前已经训练的epoch数
        validation_losses = validate_model()
        this_validation_loss = numpy.mean(validation_losses)
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    #if (this_validation_loss < best_validation_loss *
                        #improvement_threshold):
                        #patience = max(patience, iter * patience_increase)

            # save best validation score and iteration number
            best_validation_loss = this_validation_loss
            best_iter = epoch
            # test it on the test set
            test_losses = test_model()
            test_score = numpy.mean(test_losses)
        result_file.write('{0}\t{1}\n'.format(epoch, this_validation_loss * 100))    
        if epoch % 100 == 0:
            print >> stderr, epoch
            result_file.flush()
    end_time = time.clock()
    result_file.close()
    
    print(('Optimization complete with best validation score of %f %%,' \
           'with test performance %f %%''  in epoch %d') %
                 (best_validation_loss * 100., test_score * 100., best_iter))

def get_drop_info(args):
    if args.nodrop:
        return 'no-dropout'
    elif args.drop_type == 'bernoulli':
        return "_".join([args.act_type, args.drop_type, str(args.p)])
    elif args.drop_type == 'gaussian':
        info = "_".join([args.act_type, args.drop_type, str(args.mu), str(args.sigma)])
        if args.noclip:
            info = "_".join([info, 'noclip'])
        return info
    elif args.drop_type == 'uniform':
        return "_".join([args.act_type, args.drop_type, str(args.a), str(args.b)])
    else:
        print >> stderr, 'Unknown dropout type'
        assert False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-drop_type", type=str, default='bernoulli',
            help='dropout type(bernoulli, gaussian, uniform)')
    parser.add_argument("-act_type", type=str, default='sigmoid',
            help='activation function type(sigmoid, relu, tanh)')
    parser.add_argument("-p", type=float, default=0.5, 
            help='dropout probability for Bernoulli dropout')
    parser.add_argument("-mu", type=float, default=0.5, 
            help='mu used to sample Gaussian random numbers')
    parser.add_argument("-sigma", type=float, default=0.25, 
            help='sigma used to sample Gaussian random numbers')
    parser.add_argument("-a", type=float, default=0, 
            help='a used to sample Uniform random numbers')
    parser.add_argument("-b", type=float, default=1, 
            help='b used to sample Uniform random numbers')
    parser.add_argument("--nodrop", default=False, action='store_true', 
            help='if we do dropout')
    parser.add_argument("--noclip", default=False, action='store_true', 
            help='if we not clip gaussian to be in [0, 1]')
    parser.add_argument("-num_runs", default=1, type=int, 
            help='number of indepent runs')
    args = parser.parse_args()
    dataset = 'data/mnist_batches.npz'
    epoches = 3000
    dropout = False if args.nodrop else True
    for i in xrange(args.num_runs):
        save_file = 'result/' + "_".join([get_drop_info(args), str(i)]) 
        test_DBN(layers = [800,800], dropout = dropout, 
                results_file_name = save_file, seed = None, 
                training_epochs = epoches, dataset = dataset,
                dropout_args = args)


if __name__ == '__main__':
    main()
           





