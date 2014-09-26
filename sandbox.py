import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d as nconv
from theano.tensor.signal.conv import conv2d as sconv
from logistic_sgd import LogisticRegression


def classifier(Z,Zinit):
    y = T.ivector('y')

    layer3 = LogisticRegression(input=Z.reshape((Zinit[0],np.asarray(Zinit[1:]).prod())), n_in=np.asarray(Zinit[1:]).prod(), n_out=10)

    # the cost we minimize during training is the NLL of the model
    neglog = layer3.negative_log_likelihood(y)
    cerror =  layer3.errors(y)
    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params

    return neglog, cerror, params, Z, y

def testC(z):
    Z = T.tensor4('Z')
    neglog, cerror, params, Z, y =  classifier(Z,z.shape)
    f = theano.function([Z,y], neglog)
    return f(z,range(10))

def ks(I,k,axis):
        nfmaps = I.shape[axis]
        npixels = np.asarray(I.shape[axis+1:]).prod(dtype=int)
        print 'npixels', npixels
        nsamples = np.asarray(I.shape[:axis]).prod(dtype=int)
        print 'nsamples', nsamples
        lsample = npixels * nfmaps

        indices = np.zeros(nfmaps,dtype=np.int_)

        H = np.zeros(I.size)

        for s in range(nsamples):
            for p in range(npixels):
                
                for fm in range(nfmaps):
                    indices[fm] = s * lsample + p + fm * npixels

                sortinds = np.argsort(-(I.reshape(I.size)[indices]))
                indices = indices[sortinds]

                H[indices[:k]] = I.reshape(I.size)[indices[:k]]
        
        return H.reshape(I.shape)