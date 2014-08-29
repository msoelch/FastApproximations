import numpy as np
import theano
import theano.tensor as T
import gzip, cPickle
from osdfutils import crino
from theano.tensor.nnet.conv import conv2d as nconv
from theano.tensor.signal.conv import conv2d as sconv


def sh(x, theta):
    """Simple shrinkage function.
    """
    return T.sgn(x) * T.maximum(0, T.abs_(x) - theta)



def lconvista(config, shrinkage):
    """Learned Convolutional ISTA   

    Returns TODO
    """
    print "[LConvISTA]"

    layers = config['layers']
    sp_lmbd = config['lambda']
      
    _D = config['D']
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX), borrow=True, name='D')
    _theta = config['theta']
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX), borrow=True, name="theta")
    _L = config['L']
    L = theano.shared(value=np.asarray(_L, dtype=theano.config.floatX), borrow=True, name="L")
    params = (D,theta,L)

    #need tensor4:s due to interface of nnet.conv.conv2d
    #X.shape = (btsz, singleton dimension, h, w)
    X = T.tensor4('X')
    #Z.shape = (btsz, n_filters, h+s-1, w+s-1)
    Z = T.zeros(config['Zinit'])

    def rec_error(X,Z,D):
        rec = nconv(Z,D,border_mode='valid',image_shape=config['Zinit'],filter_shape=_D.shape) #rec.shape == X.shape
        rec_error = T.mean(theano.map(lambda Xi, reci: T.sum((Xi - reci)**2) , sequences=[X,rec])[0]) #alternative: T.mean( T.sum( T.sum( (x - rec)**2 , axis=3 ) , axis=2 ) )
        return rec_error


#################### TODO
    for i in range(layers):
        gradZ = nconv(nconv(Z,D,border_mode='valid',image_shape=config['Zinit'],filter_shape=_D.shape) - X , D[:,:,::-1,::-1].dimshuffle(1,0,2,3), border_mode='full')
        Z = shrinkage(Z - 1/L * gradZ, theta)
        
        
        #Z = shrinkage(Z - 1/L * T.grad(rec_error(X,Z,D),Z), theta)




    
    sparsity = T.mean(theano.map(lambda Zi: T.sum(T.abs_(Zi)),Z)[0])#alternative: mean also over filter index dimension (axis 1), not just axis 0
    cost = rec_error(X,Z,D) + sp_lmbd * sparsity

    return X, params, cost, Z



if __name__ == '__main__':
    ####################
    # READ AND FORMAT DATA
    ####################    
    mnist_f = gzip.open("mnist.pkl.gz",'rb')                              
    train_set, valid_set, test_set = cPickle.load(mnist_f)                

    dm = train_set[0].mean(axis=0)

    data = (train_set[0] - dm).reshape(train_set[0].shape[0],1,28,28)
    dtrgts = train_set[1]
    valid = (valid_set[0] - dm).reshape(valid_set[0].shape[0],1,28,28)
    vtrgts = valid_set[1]
    test = (test_set[0] - dm).reshape(test_set[0].shape[0],1,28,28)
    ttrgts = test_set[1]

    mnist_f.close()
   
     
    ####################
    # SET LEARNING PARAMETERS
    ####################  
    epochs = 20
    btsz = 100
    lr = 1
    momentum = 0.9
    decay = 0.95
    batches = data.shape[0]/btsz
    print "LConvISTA -- Learned Convolutional ISTA"
    print "Epochs", epochs
    print "Batches per epoch", batches
    print


    ####################
    # INITIALIZE ALGORITHM PARAMETERS
    #################### 

    n_filters = 100
    filter_size = 3 #filters assumed to be square


    Dinit = {"shape": (n_filters,filter_size,filter_size), "variant": "normal", "std": 0.1}
    D = crino.initweight(**Dinit)#np.random.randn(n_filters*s,s)#
    # normalize atoms (SQUARE patches) to 1
    D /= np.sqrt(np.asarray(map(lambda patch: (patch**2).sum(keepdims=True),D)))
    # reshape to four dimensions to match nnet.conv.conv2d
    D = D.reshape(1,n_filters,filter_size,filter_size)
    
    theta = 0.1 # potentially adjust to size of Z, keep it simple in the beginning

    L = 1.
    lmbd = 1. # sparsity weight
    
    zinit_shape = (btsz,n_filters,data[0,0].shape[0]+filter_size-1,data[0,0].shape[1]+filter_size-1)


    layers = 10

    config = {"D": D, "theta": theta, "L": L, "lambda": lmbd, "Zinit": zinit_shape, "layers": layers}
    
    
    
    
    
    
    
    # normalize weights according to this config
    norm_dic = {"D": {"axis":1, "c": 1.}}
    # threshold theta should be at least some value
    thresh_dic = {"theta": {"thresh": 1e-2}}

    x, params, cost, z = lconvista(config=config, shrinkage=crino.sh)
    grads = T.grad(cost, params)

    # training ...
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    # ... with stochastic gradient + momentum
    #updates = crino.momntm(params, grads, settings)#, **norm_dic)
    updates = crino.adadelta(params, grads, settings)#, **norm_dic)
    # ... normalize weights


    updates[params[0]] = updates[params[0]] / T.sqrt(theano.map(lambda patch: (patch**2).sum(keepdims=True),updates[params[0]])[0])
    #updates = crino.norm_updt(params, updates, todo=norm_dic)


    # ... make sure threshold is big enough
    updates = crino.max_updt(params, updates, todo=thresh_dic)

    train = theano.function([x], cost, updates=updates, allow_input_downcast=True)


    print 'Graph built. Learning...'


    sz = data.shape[0]
    for epoch in xrange(epochs):
        cost = 0
        for mbi in xrange(batches):
            print "mbi", mbi
            cost += btsz*train(data[mbi*btsz:(mbi+1)*btsz])
        print epoch, cost/sz