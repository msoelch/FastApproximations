import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d as conv
from mlp import LogisticRegression as LR
import gzip, cPickle
import time
from osdfutils import crino


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
    btsz = config['btsz']
    
      
    _D = config['D']
    D = theano.shared(value=np.asarray(_D, 
            dtype=theano.config.floatX),borrow=True,name='D')
    _theta = config['theta']
    theta = theano.shared(value=np.asarray(_theta, 
            dtype=theano.config.floatX),borrow=True,name="theta")
    _L = config['L']
    L = theano.shared(value=np.asarray(_L, 
            dtype=theano.config.floatX),borrow=True,name="L")

    params = [D,theta,L]

    #filter shape information for speed up of convolution
    fs1 = _D.shape
    fs2 = (fs1[1],fs1[0],fs1[2],fs1[3])

    #need tensor4:s due to interface of nnet.conv.conv2d
    #X.shape = (btsz, singleton dimension, h, w)
    X = T.tensor4('X', dtype=theano.config.floatX)
    #Z.shape = (btsz, n_filters, h+s-1, w+s-1)
    Z = T.zeros(config['Zinit'],dtype=theano.config.floatX)




    # The combination of for loop and 'hand calculated' gradient was tested
    # on CPU with 2 layers and 16 filters as well as 5 layers and 49 filters.
    # Note though that T.grad catches up with increasing parameters.
    # Hand calculated grad is preferred due to higher flexibility. 
    for i in range(layers):    
        gradZ = conv(
                    conv(
                            Z,D,border_mode='valid',
                            image_shape=config['Zinit'],filter_shape=fs1
                        ) - X,
                    D[:,:,::-1,::-1].dimshuffle(1,0,2,3),border_mode='full',
                    image_shape=config['Xshape'], filter_shape=fs2
                    ) / btsz
        Z = shrinkage(Z - 1/L * gradZ,theta)


    def rec_error(X,Z,D):
        # Calculates the reconstruction rec_i = sum_j Z_ij * D_j
        # and the corresponding (mean) square reconstruction error
        # rec_error = (X_i - rec_i) ** 2

        rec = conv(Z,D,border_mode='valid',
                    image_shape=config['Zinit'],filter_shape=_D.shape) 
        rec_error = 0.5*T.mean(((X-rec)**2).sum(axis=-1).sum(axis=-1))
        return rec_error, rec


    sparsity = T.mean(T.sum(T.sum(T.abs_(Z),axis=-1),axis=-1))
    re, rec = rec_error(X,Z,D)    

    return X, params, Z, rec, re, sparsity




def classifier(config, shrinkage):
    ####################
    # LCONVISTA LAYER
    ####################  
    X, params, Z, rec, re, sparsity = lconvista(config,shrinkage)
    y = T.ivector('y')

    ####################
    # INTERMEDIATE CNN LAYER
    #################### 
    _WCNN = config['WCNN']
    WCNN = theano.shared(value=np.asarray(_WCNN, 
            dtype=theano.config.floatX),borrow=True,name='WCNN')
    header_params = [WCNN]

    H = conv(Z,WCNN,border_mode='full')

    ####################
    # LOGISTIC REGRESSION LAYER
    ####################  
    btsz = config['btsz']
    hiddenUnits = _WCNN.shape[0] * (config['Zinit'][-2] + _WCNN.shape[2] - 1) * (config['Zinit'][3] + _WCNN.shape[3] - 1)#np.asarray(config['Zinit'][1:]).prod(dtype=np.int32)
    lrLayer = LR(input=H.reshape(shape=(btsz,hiddenUnits)),
                                            n_in=hiddenUnits, n_out=10)
    neglog = lrLayer.negative_log_likelihood(y)
    cerror = lrLayer.errors(y)
    header_params = header_params + lrLayer.params

    ####################
    # TOTAL CLASSIFIER ERROR
    ####################  
    err_weights = config["err_weights"]
    costFE = err_weights[0] * re + err_weights[1] * sparsity 
    costCL = err_weights[2] * neglog

    return X, y, params, header_params, Z, rec, costFE, costCL, cerror, re, sparsity, neglog



def runClassifier():
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
    lr = 1.
    momentum = 0.9
    decay = 0.95
    batches = data.shape[0]/btsz
    tbatches = test.shape[0]/btsz
    vbatches = valid.shape[0]/btsz
    print "LConvISTA -- Learned Convolutional ISTA"
    print "Epochs", epochs
    print "Batches per epoch", batches
    print

    ####################
    # INITIALIZE ALGORITHM PARAMETERS
    #################### 
    n_filters = 16
    filter_size = 5 #filters assumed to be square
    layers = 2
    weight_reconstruction = 0.01
    weight_sparsity = 0.01
    weight_neglog = 1

    err_weights = (weight_reconstruction,weight_sparsity,weight_neglog)

    
    Dinit = {"shape": (1,n_filters,filter_size,filter_size),
             "variant": "normal", "std": 0.5}
    D = crino.initweight(**Dinit)#np.random.randn(n_filters*s,s)#
    # normalize atoms (SQUARE patches) to 1
    D /= np.sqrt((D**2).sum(axis=-2,keepdims=True).sum(axis=-1,keepdims=True))
    
    theta = 0.001

    L = 1.

    zinit_shape = (btsz,n_filters,
                   data[0,0].shape[0]+filter_size-1,
                   data[0,0].shape[1]+filter_size-1)
    Xshape = data[:btsz].shape


    filter_size_W = 7
    n_output_filters = np.round(n_filters/10.)
    Winit =  {"shape": (n_output_filters,n_filters,filter_size_W,filter_size_W), "variant": "normal", "std": 0.5}
    W = crino.initweight(**Winit)
    W /= np.sqrt((W**2).sum(axis=-2,keepdims=True).sum(axis=-1,keepdims=True))
    


    config = {"D": D, "theta": theta, "L": L, "Zinit": zinit_shape,
              "layers": layers, "Xshape": Xshape, "btsz": btsz,
              "err_weights": err_weights, "WCNN": W}
    
    # normalize weights according to this config
    norm_dic = {"D": {"axis":1, "c": 1.}}
    # threshold theta should be at least some value
    thresh_dic = {"theta": {"thresh": 1e-15}}


    ####################
    # BUILDING GRAPH
    #################### 
    X, y, params, header_params, Z, rec, costFE, costCL, cerror, re, sparsity, neglog = classifier(config=config, shrinkage=sh)
    grads = T.grad(costFE, params)
    header_grads = T.grad(costCL, header_params)

    # training ...
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    # ... with stochastic gradient + momentum
    #updates = crino.momntm(params, grads, settings)#, **norm_dic)
    updates = crino.adadelta(params, grads, settings)#, **norm_dic)
    # ... normalize weights

    updates[params[0]] = updates[params[0]] / T.sqrt(theano.map(lambda patch: (patch**2).sum(keepdims=True),updates[params[0]][0])[0])

    # ... make sure threshold is big enough
    updates = crino.max_updt(params, updates, todo=thresh_dic)



    header_updates = crino.adadelta(header_params, header_grads, settings)



    trainFE = theano.function([X],
                (costFE, re, sparsity) + tuple(params),
                updates=updates, allow_input_downcast=True)

    trainCL = theano.function([X,y],
                (costCL, neglog) + tuple(header_params),
                updates=header_updates, allow_input_downcast=True)

    validate_or_test_model = theano.function([X,y],
                cerror, allow_input_downcast=True)
    print 'Graph built.'


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = [None] * 6
    params = [None] * 6
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < epochs) and (not done_looping):
        start = time.clock()
        epoch += 1
        for mbi in xrange(batches):
            newcost, cerror, re, sparsity, neglog, params[0], params[1], params[2], params[3], params[4], params[5] = train(data[mbi*btsz:(mbi+1)*btsz],dtrgts[mbi*btsz:(mbi+1)*btsz])
            cost += btsz*newcost
            # iteration number
            iter = (epoch - 1) * batches + mbi

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_or_test_model(valid[i*btsz:(i+1)*btsz],vtrgts[i*btsz:(i+1)*btsz]) for i in xrange(vbatches)]
                this_validation_loss = np.mean(validation_losses)
                print re, sparsity, neglog
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, mbi + 1, batches, this_validation_loss * 100.))


                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        best_params = params
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [validate_or_test_model(test[i*btsz:(i+1)*btsz],ttrgts[i*btsz:(i+1)*btsz]) for i in xrange(tbatches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, mbi + 1, batches,
                           test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    break
        end = time.clock()
        print "Elapsed time for last epoch", end-start, "seconds"
            
            
    return best_params



if __name__ == '__main__':
    runClassifier()