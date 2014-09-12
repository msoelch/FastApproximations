import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d as conv
import gzip, cPickle
import time
from osdfutils import crino
import os

from convolutional_mlp import HiddenLayer, LeNetConvPoolLayer, LogisticRegression

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
   # btsz = theano.shared(value=np.asarray(config['btsz'],dtype=theano.config.floatX), borrow=True, name='btsz')
      
    _D = config['D']
    D = theano.shared(value=np.asarray(_D, 
            dtype=theano.config.floatX),borrow=True,name='D')
    _D180 = config['D180']
    D180 = theano.shared(value=np.asarray(_D180, 
            dtype=theano.config.floatX),borrow=True,name='D180')
    _theta = config['theta']
    theta = theano.shared(value=np.asarray(_theta, 
            dtype=theano.config.floatX),borrow=True,name="theta")
    _L = config['L']
    L = theano.shared(value=np.asarray(_L, 
            dtype=theano.config.floatX),borrow=True,name="L")

    params = [D,D180,theta,L]
        
    #filter shape information for speed up of convolution
    fs1 = _D.shape
    fs2 = _D180.shape#(fs1[1],fs1[0],fs1[2],fs1[3])

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
                    D180,border_mode='full',
                    image_shape=config['Xshape'], filter_shape=fs2
                    )# / btsz
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



def trainFE(l,f,e):
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
    epochs = e
    btsz = 100
    lr = .01
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
    n_filters = f
    filter_size = 5 #filters assumed to be square
    layers = l
    w_re = 1.
    w_sparsity = .1
    print('%i filters of size %i x %i, %i layers' % (n_filters, filter_size, filter_size, layers))

    
    
    Dinit = {"shape": (1,n_filters,filter_size,filter_size),
             "variant": "normal", "std": 0.5}
    D = crino.initweight(**Dinit)#np.random.randn(n_filters*s,s)#
    # normalize atoms (SQUARE patches) to 1
    D /= np.sqrt((D**2).sum(axis=-2,keepdims=True).sum(axis=-1,keepdims=True))

    D180 = D[:,:,::-1,::-1].swapaxes(0,1)#D[:,:,::-1,::-1].dimshuffle(1,0,2,3)
    L = 15.#25.#1.    
    theta = w_sparsity / L#0.001



    zinit_shape = (btsz,n_filters,
                   data[0,0].shape[0]+filter_size-1,
                   data[0,0].shape[1]+filter_size-1)
    Xshape = data[:btsz].shape


    config = {"D": D, "D180": D180, "theta": theta, "L": L, "Zinit": zinit_shape,
              "layers": layers, "Xshape": Xshape, "btsz": btsz}
    
    # normalize weights according to this config
    norm_dic = {"D": {"axis":1, "c": 1.}}
    # threshold theta should be at least some value
    thresh_dic = {"theta": {"thresh": 1e-15}}



    ####################
    # BUILDING GRAPH
    #################### 
    X, params, Z, rec, re, sparsity = lconvista(config,sh)
    cost = w_re * re + w_sparsity * sparsity
    grads = T.grad(cost,params)

    # training ...
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    # ... with stochastic gradient + momentum
    #updates = crino.momntm(params, grads, settings)#, **norm_dic)
    updates = crino.adadelta(params, grads, settings)#, **norm_dic)
    # ... normalize weights

    updates[params[0]] = updates[params[0]] / T.sqrt(theano.map(lambda patch: (patch**2).sum(keepdims=True),updates[params[0]][0])[0])
    updates[params[1]] = updates[params[1]] / T.sqrt(theano.map(lambda patch: (patch**2).sum(keepdims=True),updates[params[1]][0])[0])


    # ... make sure threshold is big enough
    updates = crino.max_updt(params, updates, todo=thresh_dic)


    train = theano.function([X], (cost, Z, re, sparsity) + tuple(params), updates=updates,allow_input_downcast=True)
    validate = theano.function([X], (cost, re, sparsity), allow_input_downcast=True)
    extract = theano.function([X], Z, allow_input_downcast=True)

    print 'Graph built.'

    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 2500  # look as this many examples regardless
    patience_increase = 1.6  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    
    params = [None] * 4
    best_params = params

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < epochs) and (not done_looping):
        print '-----------'
        start = time.clock()
        epoch += 1
        cost = 0
        for mbi in xrange(batches):
            cost, Z, re, sparsity, params[0], params[1], params[2], params[3] = train(data[mbi*btsz:(mbi+1)*btsz])
            
            # iteration number
            iter = (epoch - 1) * batches + mbi

            if (iter + 1) % validation_frequency == 0:
                print np.asarray(params[3]), np.asarray(params[2])
                print('epoch %i, minibatch %i/%i, training set reconstruction error %f, sparsity %f, total error %f' % (epoch, mbi + 1, batches, re, sparsity, cost))
                # compute zero-one loss on validation set
                validation_losses = [validate(valid[i*btsz:(i+1)*btsz]) for i in xrange(vbatches)]
                this_validation_loss = np.mean(np.asarray(validation_losses),axis=0)


                print('epoch %i, minibatch %i/%i, validation set reconstruction error %f, sparsity %f, total error %f' % (epoch, mbi + 1, batches, this_validation_loss[1],this_validation_loss[2],this_validation_loss[0]))


                if this_validation_loss[0] < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss[0] < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_params = params
                    best_validation_loss = this_validation_loss[0]
                    best_iter = iter

                    # test it on the test set
                    test_losses = [validate(test[i*btsz:(i+1)*btsz]) for i in xrange(tbatches)]
                    test_score = np.mean(np.asarray(test_losses),axis=0)

                    print('     epoch %i, minibatch %i/%i, validation set reconstruction error %f, sparsity %f, total error %f' % (epoch, mbi + 1, batches, test_score[1],test_score[2],test_score[0]))

            if patience <= iter:
                    done_looping = True
                    break
        end = time.clock()
        print "Elapsed time for last epoch", end-start, "seconds"
    


    
            
    name = 'epochs%i_layers%i_filters%i_shape%i' % (epochs,layers,n_filters,filter_size)
    directory = "experiments/"+name
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = gzip.open("experiments/"+name+"/sparse.pkl.gz",'wb')
    cPickle.dump(best_params, f)
                
            
    return best_params,Z



def testClassification():
    layers = 10
    n_filters = 16
    filter_size = 5

    ####################
    # READ AND FORMAT DATA
    ####################  

    mnist_f = gzip.open("mnist.pkl.gz",'rb')                              
    train_set, valid_set, test_set = cPickle.load(mnist_f)                

    dm = train_set[0].mean(axis=0)

    data = (train_set[0][::2] - dm).reshape(train_set[0][::2].shape[0],1,28,28)
    dtrgts = train_set[1][::2]
    n_tr_samples = dtrgts.size

    valid = (valid_set[0][::2] - dm).reshape(valid_set[0][::2].shape[0],1,28,28)
    vtrgts = valid_set[1][::2]
    n_va_samples = vtrgts.size

    test = (test_set[0][::2] - dm).reshape(test_set[0][::2].shape[0],1,28,28)
    ttrgts = test_set[1][::2]
    n_te_samples = ttrgts.size

    x = np.concatenate((data,valid,test))

    mnist_f.close()


    file = "experiments/epochs50_layers10_filters16_shape5/sparse.pkl.gz"  
    print file
    f = gzip.open(file,'rb')                              
    params = cPickle.load(f)                
    f.close()

    D = np.asarray(params[0].astype(np.float32),dtype=theano.config.floatX)
    D180 = np.asarray(params[1].astype(np.float32),dtype=theano.config.floatX)
    theta = np.asarray(params[2].astype(np.float32),dtype=theano.config.floatX)
    L = np.asarray(params[3].astype(np.float32),dtype=theano.config.floatX)

    zinit_shape = (x.shape[0],n_filters,
                   data[0,0].shape[0]+filter_size-1,
                   data[0,0].shape[1]+filter_size-1)

    btsz = 25000

    config = {"D": D, "D180": D180, "theta": theta, "L": L, "Zinit": zinit_shape,
              "layers": layers, "Xshape": x.shape, "btsz": btsz}#x.shape[0]}

    X, params, Z, rec, re, sparsity = lconvista(config,sh)

    extract = theano.function([X], Z, allow_input_downcast=True)

    allZ = extract(x)
    print x.shape
    print allZ.shape
    print np.count_nonzero(allZ)

    trainZ = allZ[:n_tr_samples]
    validZ = allZ[n_tr_samples:n_tr_samples+n_va_samples]
    testZ = allZ[n_tr_samples+n_va_samples:]
    

    btsz = 100
    batches = data.shape[0]/btsz
    tbatches = test.shape[0]/btsz
    vbatches = valid.shape[0]/btsz
    Zinit =  (btsz,) + trainZ.shape[1:]

    neglog, cerror, cparams, Z, y = classifier(Zinit)
    grads = T.grad(neglog,cparams)
    # training ...
    lr = 1.
    momentum = 0.9
    decay = 0.95
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    # ... with stochastic gradient + momentum
    #updates = crino.momntm(params, grads, settings)#, **norm_dic)
    updates = crino.adadelta(cparams, grads, settings)#, **norm_dic)


    train = theano.function([Z,y], (neglog,cerror), updates=updates, allow_input_downcast=True)
    validate = theano.function([Z,y], (neglog,cerror), allow_input_downcast=True)
    



    print '... training'

    # early-stopping parameters
    patience = 2500  # look as this many examples regardless
    patience_increase = 1.6  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    
    #params = [None] * 4
    #best_params = params

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epochs = 20
    epoch = 0
    done_looping = False

    while (epoch < epochs) and (not done_looping):
        print '-----------'
        start = time.clock()
        epoch += 1
        cost = 0
        for mbi in xrange(batches):
            neglog, cerror = train(trainZ[mbi*btsz:(mbi+1)*btsz], dtrgts[mbi*btsz:(mbi+1)*btsz])
            
            # iteration number
            iter = (epoch - 1) * batches + mbi

            if (iter + 1) % validation_frequency == 0:
                print('epoch %i, minibatch %i/%i, training set neglog %f, classification error %f' % (epoch, mbi + 1, batches, neglog,cerror))
                # compute zero-one loss on validation set
                validation_losses = [validate(validZ[i*btsz:(i+1)*btsz],vtrgts[i*btsz:(i+1)*btsz]) for i in xrange(vbatches)]
                this_validation_loss = np.mean(np.asarray(validation_losses),axis=0)


                print('epoch %i, minibatch %i/%i, validation set neglog %f, classification error %f' % (epoch, mbi + 1, batches, this_validation_loss[0],this_validation_loss[1]))


                if this_validation_loss[0] < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss[0] < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss[0]
                    best_iter = iter

                    # test it on the test set
                    test_losses = [validate(testZ[i*btsz:(i+1)*btsz],ttrgts[i*btsz:(i+1)*btsz]) for i in xrange(tbatches)]
                    test_score = np.mean(np.asarray(test_losses),axis=0)

                    print('     epoch %i, minibatch %i/%i, test set neglog %f, classification error %f' % (epoch, mbi + 1, batches, test_score[0] , test_score[1]))

            if patience <= iter:
                    done_looping = True
                    break
        end = time.clock()
        print "Elapsed time for last epoch", end-start, "seconds"
    



def classifier(Zinit):
    Z = T.tensor4('Z')
    y = T.ivector('y')
    rng = np.random.RandomState()

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-5+1,32-5+1)=(28,28)
    # maxpooling reduces this further to (28/2,28/2) = (14,14)
    # 4D output tensor is thus of shape (btsz,firstfilters,14,14) 100.20.14.14
    firstfilters = 20
    layer0 = LeNetConvPoolLayer(rng, Z,
            image_shape=Zinit,
            filter_shape=(firstfilters, Zinit[1], 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (14 - 5 + 1, 14 - 5 + 1)=(10, 10)
    # maxpooling reduces this further to (10/2,10/2) = (5, 5)
    # 4D output tensor is thus of shape (btsz,50,5,5)
    secondfilters=50    
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(Zinit[0], firstfilters, 14, 14),
            filter_shape=(secondfilters, firstfilters, 5, 5), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (btsz, 50*5*5) = (100, 1250)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input,
                         n_in=1250, n_out=500,
                         activation=T.tanh    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    neglog = layer3.negative_log_likelihood(y)
    cerror =  layer3.errors(y)
    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    return neglog, cerror, params, Z, y

"""

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
    n_filters = 25
    filter_size = 5 #filters assumed to be square
    layers = 10
    weight_reconstruction = 1.
    weight_sparsity = 1.
    weight_neglog = 1.

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
    grads = T.grad(costFE,params)
    grads += T.grad(costCL,header_params)
    #grads = T.grad(costFE+costCL, params+header_params)
    

    # training ...
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    # ... with stochastic gradient + momentum
    #updates = crino.momntm(params, grads, settings)#, **norm_dic)
    updates = crino.adadelta(params+header_params, grads, settings)#, **norm_dic)
    # ... normalize weights

    updates[params[0]] = updates[params[0]] / T.sqrt(theano.map(lambda patch: (patch**2).sum(keepdims=True),updates[params[0]][0])[0])

    # ... make sure threshold is big enough
    updates = crino.max_updt(params, updates, todo=thresh_dic)



    #header_updates = crino.adadelta(header_params, header_grads, settings)

    train = theano.function([X,y], (costFE, re, sparsity) + tuple(params) + (costCL, neglog), updates=updates,allow_input_downcast=True)


    #trainFE = theano.function([X],
    #            (costFE, re, sparsity) + tuple(params),
    #            updates=updates, allow_input_downcast=True)

    #trainCL = theano.function([X,y],
    #            (costCL, neglog) + tuple(header_params),
    #            updates=header_updates, allow_input_downcast=True)

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

    
    params = [None] * 3
    best_params = params

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < epochs) and (not done_looping):
        start = time.clock()
        epoch += 1
        #FE = (epoch % 2 == 1) # feature extraction
        #if FE:
        #    print 'Training feature extraction'
        #else:
        #    print 'Training classification'
        cost = 0
        for mbi in xrange(batches):

            #if FE:
            #    costFE, re, sparsity, params[0], params[1], params[2] = trainFE(data[mbi*btsz:(mbi+1)*btsz])
            #    cost += btsz*costFE
            #else:
            #    costCL, neglog, header_params[0], header_params[1], header_params[2] = trainCL(data[mbi*btsz:(mbi+1)*btsz],dtrgts[mbi*btsz:(mbi+1)*btsz])
            #    cost += btsz*costCL

            costFE, re, sparsity, params[0], params[1], params[2], costCL, neglog = train(data[mbi*btsz:(mbi+1)*btsz],dtrgts[mbi*btsz:(mbi+1)*btsz])
            cost += btsz*(costFE+costCL)

            # iteration number
            iter = (epoch - 1) * batches + mbi

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_or_test_model(valid[i*btsz:(i+1)*btsz],vtrgts[i*btsz:(i+1)*btsz]) for i in xrange(vbatches)]
                this_validation_loss = np.mean(validation_losses)
                #if FE:
                #    print re, sparsity
                #else:
                #    print neglog
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

"""

if __name__ == '__main__':
    testClassification()

