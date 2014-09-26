"""
Some theano offspring.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv as Tconv
import gzip, cPickle





def lbcofb(config, shrinkage):
    """Learned version of BCoFB by Sprecheman, Bronstein, Sapiro.

    Returns TODO
    """
    layers = config['layers']
    sp_lmbd = config['lambda']
    sp_mu = config['mu']
    L = config['L']
    Dinit = config['D']
    partition =  config['partition'] # holds the indices of all first group elements (plus very last for avoiding index errors)

    x = T.matrix('x')
    #partition = T.ivector('partition')
    #lower = partition[:-1]
    #upper = partition[1:]
    #lower = T.ivector('lower')#partition[:-1]
    #upper = T.ivector('upper')#partition[1:]
    #n_groups = lower.shape[0]
    
    
    lower = theano.shared(value=partition[:-1], borrow=True, name="lower")
    upper = theano.shared(value=partition[1:], borrow=True, name="upper")
    n_groups = len(partition) - 1
    
    # D for dictionary
    _D = initweight(**Dinit)
    # normalize atoms of dictionary (== rows) to length 1
    _d = np.sqrt(np.sum(_D * _D, axis=1))
    _D /= np.atleast_2d(_d).T
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
            borrow=True, name='D')

    _S = np.eye(_D.shape[0]) - 1./L * np.dot(_D ,_D.T)
    S = theano.shared(value=np.asarray(_S, dtype=theano.config.floatX),
            borrow=True, name='S')
    
    _theta = np.abs(np.random.randn(_S.shape[0],))
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX),
            borrow=True, name="theta")
    
    _grtheta = np.abs(np.random.randn(n_groups,))
    grtheta = theano.shared(value=np.asarray(_grtheta, dtype=theano.config.floatX),
            borrow=True, name="grtheta")

    L = theano.shared(value=np.asarray(L, dtype=theano.config.floatX),
            borrow=True, name="L")


    params = (D, S, theta, grtheta, L)
    

    #NS number of test samples
    #NA number of atoms (rows of D)
    #I image/atome dimension
    #NG number of groups
    #R size of specific group (flexible)    
    b = T.dot(x, D.T) #NSxI * IxNA = NSxNA
    z = T.zeros(b.shape) #NSxNA
    norms = T.zeros((x.shape[0],n_groups)) #NSxNG
    for i in range(layers):
        y = sh(sh(b,theta),grtheta,partition) #NSxNA
        e = y - z #NSxNA
        norms, n_updt = theano.map(maxn, sequences=[lower,upper], non_sequences=[e])
        g = T.argmax(norms,axis=1) #NS
        deltab, _b_updt = theano.map(subdot, sequences=[g, e], non_sequences=[lower, upper, S])
        b = b + deltab
        z, _z_updates = theano.map(fn=lambda z,y,g,lower,upper: T.set_subtensor( z[lower[g]:upper[g]] , y[lower[g]:upper[g]]),
                                         sequences=[z,y,g], non_sequences=[lower,upper])
    z = sh(sh(b,theta),grtheta,partition)
  

    rec = T.dot(z, L * D)
    cost = T.mean(T.sum((x - rec)**2, axis=1))
    sparsity = T.mean(T.sum(T.abs_(z), axis=1))
    gr_sparsity = 0
    for (start,end) in zip(lower,upper):
        gr_sparsity = gr_sparsity + T.mean(T.sum(z[start:end] ** 2))

    cost = cost + sp_lmbd * sparsity + sp_mu * gr_sparsity
    return x, params, cost, rec, z



def sh1(x, theta, partition=None):
    """Singleton (default) and group shrinkage function.
    """
    if partition is None:
        return T.sgn(x) * T.maximum(0, T.abs_(x) - theta)
    else:
        for i, (start,end) in enumerate(zip(partition[:-1],partition[1:])): 
                xr = x[start:end]
                norm = T.sqrt(T.sum(xr**2))+1e-8
                T.set_subtensor(x[start:end], x[start:end]*T.maximum(0, norm - theta[i])/norm)
        return x
        #lower = partition[:-1]
        #upper = partition[1:]
        #x, _ = theano.map(shgr , sequences=[ lower, upper, theta], non_sequences=[x] )
        #return x


def sh2(x, theta, partition=None):
    """Singleton (default) and group shrinkage function.
    """
    if partition is None:
        return T.sgn(x) * T.maximum(0, T.abs_(x) - theta)
    else:
        lower = partition[:-1]
        upper = partition[1:]
        x, _ = theano.map(shgr , sequences=[ lower, upper, theta], non_sequences=[x] )
        return x

"""
map helper functions
"""
def shgr(lower, upper, theta, x):
        #for i, (start,end) in enumerate(zip(partition[:-1],partition[1:])): 
        #        xr = x[start:end]
        #        norm = T.sqrt(T.sum(xr**2))+1e-8
        #        T.set_subtensor(x[start:end], x[start:end]*T.maximum(0, norm - theta[i])/norm)
        #return x
    norm = T.sqrt(T.sum(x[lower:upper]**2))+1e-8
    T.set_subtensor(x[lower:upper], x[lower:upper]*T.maximum(0, norm - theta)/norm)
    return x


def maxn(idx1, idx2, x):
        #for j, (start,end) in enumerate(zip(partition[:-1],partition[1:])):
        #    T.set_subtensor(norms[:,j],T.sqrt(T.sum(e[:,start:end]**2,axis=1)))
    return T.sum(x[:, idx1:idx2]**2, axis=1)


def subdot(idx, e , idx1, idx2, S):
    return T.dot(S[:, idx1[idx]:idx2[idx]], e[idx1[idx]:idx2[idx]].T).T

"""
end map helper functions
"""

   
def initweight(shape, variant="normal", **kwargs):
    """
    Init weights.
    """
    if variant is "normal":
        if "std" in kwargs:
            std = kwargs["std"]
        else:
            std = 0.1
        weights = np.asarray(np.random.normal(loc=0, scale=std, size=shape),
                dtype=theano.config.floatX)
    elif variant is "uniform":
        units = shape[0]*shape[1]
        bound = 4*np.sqrt(6./units)
        weights = np.asarray(np.uniform(low=-bound, high=bound, size=shape),
                dtype=theano.config.floatX)
    elif variant is "sparse":
        sparsity = kwargs["sparsity"]
        weights = np.zeroes(shape, dtype=theano.config.floatX)
        for w in weights:
            w[random.sample(xrange(n), sparsity)] = np.random.randn(sparsity)
        weights = weights.T
    else:
        assert False, 'Problem in initweight.'

    return weights



def momntm(params, grads, settings, **kwargs):
    """
    Optimizer: SGD with momentum.
    """
    lr = settings['lr']
    momentum = settings['momentum']
    print "[MOMNTM] lr: {0}; momentum: {1}".format(lr, momentum)

    _moments = []
    for p in params:
        p_mom = theano.shared(np.zeros(p.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        _moments.append(p_mom)

    updates = OrderedDict()
    for grad_i, mom_i in zip(grads, _moments):
        updates[mom_i] =  momentum*mom_i + lr*grad_i

    for param_i, mom_i in zip(params, _moments):
            updates[param_i] = param_i - updates[mom_i]

    return updates


def norm_updt(params, updates, todo):
    """Normalize updates wrt length.
    """
    for p in params:
        if p.name in todo:
            axis = todo[p.name]['axis']
            const = todo[p.name]['c']
            print "[NORM_UPDT] {0} normalized to {1} along axis {2}".format(p.name, const, axis)
            wl = T.sqrt(T.sum(T.square(updates[p]), axis=axis) + 1e-6)
            if axis == 0:
                updates[p] = const * updates[p]/wl
            else:
                updates[p] = const * updates[p]/wl.dimshuffle(0, 'x')
    return updates


def max_updt(params, updates, todo):
    """Normalize updates wrt to minum value.
    """
    for p in params:
        if p.name in todo:
            thresh = todo[p.name]['thresh']
            print "[MAX_UPDT] {0} at least at {1}".format(p.name, thresh)
            updates[p] = T.maximum(thresh, updates[p])
    return updates


if __name__ == '__main__':
    mnist_f = gzip.open("mnist.pkl.gz",'rb')                              
    train_set, valid_set, test_set = cPickle.load(mnist_f)                
    data = train_set[0]
    trgts = train_set[1]
    valid = valid_set[0]
    mnist_f.close()

    epochs = 20
    btsz = 100
    enc_out = 2
    lr = 0.001
    momentum = 0.9
    decay = 0.95
    batches = data.shape[0]/btsz
    print "LBCoFB -- Learned BCoFB without ISTA"
    print "Epochs", epochs
    print "Batches per epoch", batches
    print

    sp_dim = 32*32
    Dinit = {"shape": (sp_dim, 28*28), "variant": "normal", "std": 0.1}
    n_groups = 10
    lmbd = 0.1#np.full((sp_dim,),0.1) # singleton sparsity weights
    mu = 0.1#np.full((n_groups,),0.1) # group sparsity weights
    L = 1.
    layers = 2
    partition = np.round(np.linspace(0,sp_dim,n_groups+1)).astype(int)#range(sp_dim)[::sp_dim//n_groups] # group starting indices
    lower = partition[:-1]
    upper = partition[1:]

    config = {"D": Dinit, "layers": layers, "L": L, "lambda": lmbd, "mu": mu, "partition": partition}
    # normalize weights according to this config
    norm_dic = {"D": {"axis":1, "c": 1.}}
    # threshold theta should be at least some value
    thresh_dic = {"theta": {"thresh": 1e-2}}

    x, params, cost, rec, z = lbcofb(config=config, shrinkage= sh)
    grads = T.grad(cost, params)
    zip
    # training ...
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    # ... with stochastic gradient + momentum
    updates =  momntm(params, grads, settings)#, **norm_dic)
    # ... normalize weights
    updates =  norm_updt(params, updates, todo=norm_dic)
    # ... make sure threshold is big enough
    updates =  max_updt(params, updates, todo=thresh_dic)

    train = theano.function([x, part], cost, updates=updates, 
                        allow_input_downcast=True)
    print 'done.'


    sz = data.shape[0]
    for epoch in xrange(epochs):
        cost = 0
        for mbi in xrange(batches):
            cost += btsz*train(data[mbi*btsz:(mbi+1)*btsz], lower, upper)
        print epoch, cost/sz