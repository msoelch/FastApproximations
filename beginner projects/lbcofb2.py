import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
import gzip, cPickle





def lbcofb(config, shrinkage):
    """Learned version of BCoFB by Sprecheman, Bronstein, Sapiro.
        http://icml.cc/2012/papers/332.pdf

    Returns TODO
    """
    layers = config['layers']


    _sp_lmbd = config['lambda']
    sp_lmbd = theano.shared(value=np.asarray(_sp_lmbd, dtype=theano.config.floatX),
        borrow=True, name='sp_lmbd')
    _sp_mu = config['mu']
    sp_mu = theano.shared(value=np.asarray(_sp_mu, dtype=theano.config.floatX),
        borrow=True, name='sp_mu')


    Dinit = config['D']
    _D = initweight(**Dinit)
    _d = np.sqrt(np.sum(_D * _D, axis=1)) # normalize atoms of dictionary (== rows) to length 1
    _D /= np.atleast_2d(_d).T
    D = theano.shared(value=np.asarray(_D, dtype=theano.config.floatX),
            borrow=True, name='D')


    _L = config['L']
    L = theano.shared(value=np.asarray(_L, dtype=theano.config.floatX),
            borrow=True, name="L")

    _S = np.eye(_D.shape[0]) - 1./_L * np.dot(_D ,_D.T)
    S = theano.shared(value=np.asarray(_S, dtype=theano.config.floatX),
            borrow=True, name='S')
    

    partition =  config['partition'] # holds the indices of all first group elements (plus very last for avoiding index errors)
    n_groups = len(partition) - 1
    lower = theano.shared(value=np.asarray(partition[:-1]), borrow=True, name='lower')
    upper = theano.shared(value=np.asarray(partition[1:]), borrow=True, name='upper')
    lower_singleton = T.arange(_D.shape[1])[:-1]
    upper_singleton = T.arange(_D.shape[1])[1:]


    _theta = np.abs(np.random.randn(_S.shape[0],))
    theta = theano.shared(value=np.asarray(_theta, dtype=theano.config.floatX),
            borrow=True, name="theta")
    

    _grtheta = np.abs(np.random.randn(n_groups,))
    grtheta = theano.shared(value=np.asarray(_grtheta, dtype=theano.config.floatX),
            borrow=True, name="grtheta")


    x = T.matrix('x')


    params = (D, S, theta, grtheta, L)



    # local defs
    # norm(e_r)
    def maxn(lower, upper, x):
        return T.sum(x[:, lower:upper]**2, axis=1)

    # S_g * e_g
    def subdot(g, e , lower, upper, S):
        return T.dot(e[lower[g]:upper[g]], S[:, lower[g]:upper[g]].T)
    
    # z_g = y_g
    def subtens(z,y,g,lower,upper): 
        return T.set_subtensor( z[lower[g]:upper[g]] , y[lower[g]:upper[g]])


    



    b = T.dot(x, 1/L * D.T)
    z = T.zeros(b.shape)
    s = grtheta/L
    t = theta/L
    for i in range(layers):
        y = sh(sh(b,t,lower_singleton, upper_singleton),s,lower,upper) 
        e = y - z 
        norms, _ = theano.map(maxn, sequences=[lower, upper], non_sequences=[e])
        g = T.argmax(norms,axis=0)
        deltab, _ = theano.map(subdot, sequences=[g, e], non_sequences=[lower, upper, S])
        b = b + deltab
        z, _ = theano.map(subtens, sequences=[z, y, g], non_sequences=[lower, upper])
    z = sh(sh(b,t,lower_singleton, upper_singleton),s,lower,upper)






    rec = T.dot(z, L * D)
    recerror = T.mean(T.sum((x - rec)**2, axis=1))
    sparsity = T.mean(T.sum(sp_lmbd.dimshuffle('x',0)*T.abs_(z), axis=1))
    
    _gr_sparsity, _ = theano.map(lambda lower, upper, mu, z: mu * T.sqrt(T.sum(z[:,lower:upper] ** 2, axis=1)+1e-8), sequences=[lower,upper,sp_mu], non_sequences=[z])
    gr_sparsity = T.mean(T.sum(_gr_sparsity,axis=0))
    test = L

    cost = recerror + sparsity + gr_sparsity

    return x, params, cost, rec, z, lower, upper



def sh(x, theta, lower, upper):
    #sub routine shrinking all samples within one fixed group (ranging in index interval lower:upper) by scalar theta
    def shgr(lower, upper, theta, x):
        norm = T.sqrt(T.sum(x[:,lower:upper]**2,axis=1)+1e-8)
        x = T.set_subtensor(x[:,lower:upper], (T.maximum(0, norm - theta)/norm).dimshuffle(0,'x')*x[:,lower:upper])
        return x

    #looping over all groups
    shrinked, _ = theano.scan(shgr , sequences=[lower, upper, theta], outputs_info=[x] )
    return shrinked[-1] #keep only last iteration, i.e., when all groups are included








   
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
    dm = data.mean(axis=0)
    data = data - dm
    trgts = train_set[1]
    valid = (valid_set[0] - dm)
    valid_trgts = valid_set[1]
    test = test_set[0] - dm
    test_trgts = test_set[1]

    mnist_f.close()

    epochs = 20
    btsz = 200
    enc_out = 2
    lr = 10e-3
    momentum = 0.9
    decay = 0.95
    batches = data.shape[0]/btsz
    print "LBCoFB -- Learned BCoFB"
    print "Epochs", epochs
    print "Batches per epoch", batches
    print

    sp_dim = 32*32
    Dinit = {"shape": (sp_dim, 28*28), "variant": "normal", "std": 0.1}
    n_groups = 10
    lmbd = np.full((sp_dim,),0.1) # singleton sparsity weights
    mu = np.full((n_groups,),0.1) # group sparsity weights
    L = 1.
    layers = 2
    partition = np.round(np.linspace(0,sp_dim,n_groups+1)).astype(np.int32)


    config = {"D": Dinit, "layers": layers, "L": L, "lambda": lmbd, "mu": mu, "partition": partition}
    # normalize weights according to this config
    norm_dic = {"D": {"axis":1, "c": 1.}}
    # threshold theta should be at least some value
    thresh_dic = {"theta": {"thresh": 1e-2}}

    x, params, cost, rec, z, lower, upper = lbcofb(config=config, shrinkage= sh)
    grads = T.grad(cost, params)
    
    # training ...
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    # ... with stochastic gradient + momentum
    updates =  momntm(params, grads, settings)#, **norm_dic)
    # ... normalize weights
    updates =  norm_updt(params, updates, todo=norm_dic)
    # ... make sure threshold is big enough
    updates =  max_updt(params, updates, todo=thresh_dic)

    train = theano.function([x], cost, updates=updates, 
                        allow_input_downcast=True)
    print 'done.'

    sz = data.shape[0]
    epochs = 10
    for epoch in xrange(epochs):
        cost = 0
        for mbi in xrange(batches):
            cost += btsz*train(data[mbi*btsz:(mbi+1)*btsz])
            print "minibatch {0} of {1}, epoch {2}".format(mbi,batches,epoch)
        print epoch, cost/sz
