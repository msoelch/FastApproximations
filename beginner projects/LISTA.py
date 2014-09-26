import time
import numpy as np
from logistic_sgd import LogisticRegression, load_data2
import theano
from theano import tensor as T
from helpers import visualize, _scale_01
import sys
import os
try:
    import Image as img
except:
    import PIL as img
from random import sample



dictInitializationVar = 0.01
lipschitzConstant = 1.



class LISTA(object):
    def __init__(self, X, n_in, sparse_dimension, n_layers, ts):
        
            
        W_d = np.float64(ts[sample(xrange(ts.shape[0]),sparse_dimension),:].T) + 0.5*np.random.randn(n_in,sparse_dimension)
        visualize(W_d.T,28**2).save('dict.png')
        #initializing parameters
        #W_d = dictInitializationVar*np.random.randn(n_in,sparse_dimension)
        length = np.sqrt(np.sum(np.square(W_d), axis=0) + 1e-8)
        W_d = W_d/length
        visualize(W_d.T,28**2).save('dictnormal.png')
        L = lipschitzConstant

        self.L = theano.shared(value=L,name='L')
        self.W_e = theano.shared(value=1/L*W_d.T,name='W_e')
        self.S = theano.shared(value=np.eye(sparse_dimension) - 1/L * np.dot(W_d.T,W_d) ,name='S')

        theta = np.full(sparse_dimension,0.8)
        theta = 0.1*np.random.rand(sparse_dimension)+0.65
        self.theta = theano.shared(value=theta, name='theta')

        #np.abs(2*np.random.rand(sparse_dimension)+0.5

        self.params = [self.W_e, self.S, self.theta, self.L]

        

        #encoding
        self.Z = self.fprop(X, n_layers)

        #decoding/reconstruction
        self.Xrec = T.dot(self.Z, self.L * self.W_e)
           
        #regressing sparse codes on digits       
        self.logRegressionLayer = LogisticRegression(input=self.Z, n_in=sparse_dimension, n_out=10)
        self.params += self.logRegressionLayer.params
        
        
        
        # different errors for regularization
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.sparsity = T.mean(T.abs_(self.Z).sum(axis=1))
        self.reconstructionError = T.mean(((X - self.Xrec )**2).sum(axis=1))
               
        

    # shrinkage function
    def h_theta(self, Y):
        return T.sgn(Y) * T.maximum((T.abs_(Y) - self.theta),0)

    # forward propagation, core part of LISTA
    def fprop(self, X, n_layers):
        B = T.dot(X, T.transpose(self.W_e)) 
        Z = self.h_theta(B)  
        for i in range(n_layers):
            C = B + T.dot(Z, self.S) 
            Z = self.h_theta(C) 
        return Z






##########
#
#
#
def test_LISTA(learning_rate=0.0001, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=225):
#
#
#
##########

    datasets = load_data2(dataset) 

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size


    # BUILD ACTUAL MODEL
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
       




##########
#
#
#
    # construct the LISTA class
    n_layers =  15
    classifier = LISTA(X=x,n_in=28*28,sparse_dimension=20**2,n_layers=n_layers, ts=train_set_x)
    
        
    cost = classifier.reconstructionError + 10e-1*classifier.sparsity + 10e-8*classifier.negative_log_likelihood(y) 
#
#
#
##########

            



    # compute the gradient of cost with respect to all parameters (sotred in params), the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    for param, gparam in zip(classifier.params, gparams):
        up = param - learning_rate * gparam
        if param.name=="W_e":                             ### Christian: Normalize lengths
            length = T.sqrt(T.sum(T.square(up), axis=1) + 1e-8)
            up = up/length.dimshuffle(0, 'x')             ### Christian: dimshuffle necessary to get shape right!
        updates.append((param, up))



##########
#
#
#
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[x, y],    ### Christian: use numpy arrays directly as inputs
            outputs=[classifier.Z, classifier.W_e, classifier.S, classifier.theta, classifier.Xrec], updates=updates)  ### Christian: look at output of sparse coder
#
#
#
##########







    # TRAIN MODEL 
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many minibatche before checking the network on the validation set; in this case we check every epoch

    best_params = None
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    
    epoch = 0
    done_looping = False
    n_epochs = 2   ### Christian: one epoch to get numerical stability correct.
    while (epoch < n_epochs) and (not done_looping):
        print "<<<<<<<<<<<<<<<<<"
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches): #### Christian: enough minibatches for now

            mbi = minibatch_index  ### Christian: next three lines for loading arrays into train function
            mb = np.asarray(train_set_x[mbi*batch_size:(mbi+1)*batch_size])#, dtype=theano.config.floatX)
            mbt = np.asarray(train_set_y[mbi*batch_size:(mbi+1)*batch_size], dtype=np.int32)

            tmp= train_model(mb, mbt)
            
            
        visualize(tmp[4],28*28).save("test.png")
        visualize(tmp[1],784).save("matrix.png")
        visualize(tmp[0],20**2).save("sparse.png")
        print tmp[3]
        print np.mean(tmp[3])
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))



if __name__ == '__main__':
    test_LISTA()
