import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d as nconv
from theano.tensor.signal.conv import conv2d as sconv
from logistic_sgd import LogisticRegression
import time
from ksparse import kSparse


def sh(x, theta,k):
    """Simple shrinkage function.
    """
    return T.sgn(x) * T.maximum(0, T.abs_(x) - theta)

def sh_ksparse(x, theta,k,axes):
    """Advanced shrinkage function shrinking all but the k largest entries
    """
    #   ksZ = kSparse(x,k,axes)
    ksZ = T.sgn(x)*kSparse(T.abs_(x),k,axes)
    return sh(x,theta,k) - sh(ksZ,theta,k) + ksZ


#import sys
#import time

#def restart_line():
#    sys.stdout.write('\r')
#    sys.stdout.flush()


#def write_inline():
#    sys.stdout.write('\r')
#    sys.stdout.flush()
#    sys.stdout.write(str(i))

#if __name__ == '__main__':
#    test()


#def classifier(Z,Zinit):
#    y = T.ivector('y')

#    layer3 = LogisticRegression(input=Z.reshape((Zinit[0],np.asarray(Zinit[1:]).prod())), n_in=np.asarray(Zinit[1:]).prod(), n_out=10)

#    # the cost we minimize during training is the NLL of the model
#    neglog = layer3.negative_log_likelihood(y)
#    cerror =  layer3.errors(y)
#    # create a list of all model parameters to be fit by gradient descent
#    params = layer3.params

#    return neglog, cerror, params, Z, y

#def testC(z):
#    Z = T.tensor4('Z')
#    neglog, cerror, params, Z, y =  classifier(Z,z.shape)
#    f = theano.function([Z,y], neglog)
#    return f(z,range(10))

def sh(x, theta,k):
    """Simple shrinkage function.
    """
    return T.sgn(x) * T.maximum(0, T.abs_(x) - theta)

def sh_ksparse(x, theta,k,axes):
    """Advanced shrinkage function shrinking all but the k largest entries
    """
    #   ksZ = kSparse(x,k,axes)
    ksZ = T.sgn(x)*kSparse(T.abs_(x),k,axes)
    return sh(x,theta,k) - sh(ksZ,theta,k) + ksZ

x = T.tensor4('x')
theta = T.scalar('theta')
k = T.iscalar('k')
axes = T.ivector('axes')

s = theano.function([x,theta,k,axes],sh_ksparse(x,theta,k,axes),allow_input_downcast=True)