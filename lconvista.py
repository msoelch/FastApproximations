import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d as conv
import gzip, cPickle
import time
import crino
import os
from convolutional_mlp import HiddenLayer, LeNetConvPoolLayer, LogisticRegression
from ksparse import kSparse


def sh(x, theta,k):
    """Simple shrinkage function.
    """
    return T.sgn(x) * T.maximum(0, T.abs_(x) - theta)

def sh_ksparse(x, theta,k):
    ksZ = kSparse(x,k,axis=1)
    return sh(x,theta,k) - sh(ksZ,theta,k) + ksZ

#class LConvISTA(object):
#    def __init__(self, layers, k, D, D180, theta, L):
    
#        fs1 = _D.shape
#        fs2 = _D180.shape

#        D = theano.shared(value=np.asarray(D, 
#                dtype=theano.config.floatX),borrow=True,name='D')
#        D180 = theano.shared(value=np.asarray(D180, 
#                dtype=theano.config.floatX),borrow=True,name='D180')
#        theta = theano.shared(value=np.asarray(theta, 
#                dtype=theano.config.floatX),borrow=True,name="theta")
#        L = theano.shared(value=np.asarray(L, 
#                dtype=theano.config.floatX),borrow=True,name="L")

#        self.params = [D,D180,theta,L]

#        #X.shape = (btsz, singleton dimension, h, w)
#        self.X = T.tensor4('X', dtype=theano.config.floatX)
#        #Z.shape = (btsz, n_filters, h+s-1, w+s-1)
#        Z = T.zeros(config['Zinit'],dtype=theano.config.floatX)

#        for i in range(layers):    
#            gradZ = conv(
#                        conv(
#                                Z,D,border_mode='valid',
#                                image_shape=config['Zinit'],filter_shape=fs1
#                            ) - self.X,
#                        D180,border_mode='full',
#                        image_shape=config['Xshape'], filter_shape=fs2
#                        )
#            Z = shrinkage(Z - 1/L * gradZ,theta,k)

#        self.Z = Z

#        self.rec = conv(Z,D,border_mode='valid',
#                        image_shape=config['Zinit'],filter_shape=_D.shape) 
#        self.rec_error = 0.5*T.mean(((X-rec)**2).sum(axis=-1).sum(axis=-1))


#        self.sparsity = T.mean(T.sum(T.sum(T.abs_(kSparse(Z,k,1) - Z),axis=-1),axis=-1))


def lconvista(config):
    """Learned Convolutional ISTA   

    Returns TODO
    """
    print "[LConvISTA]"

    layers = config['layers']

    k = config['k']
      
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
    fs2 = _D180.shape

    #need tensor4:s due to interface of nnet.conv.conv2d
    #X.shape = (btsz, singleton dimension, h, w)
    X = T.tensor4('X', dtype=theano.config.floatX)
    #Z.shape = (btsz, n_filters, h+s-1, w+s-1)
    Z = T.zeros(config['Zinit'],dtype=theano.config.floatX)




    # The combination of for loop and 'hand calculated' gradient was tested
    # on CPU with 2 layers and 16 filters as well as 5 layers and 49 filters.
    # Note though that T.grad catches up with increasing parameters.
    # Hand calculated grad is preferred due to higher flexibility. 

    shrinkage = sh_ksparse

    for i in range(layers):    
        gradZ = conv(
                    conv(
                            Z,D,border_mode='valid',
                            image_shape=config['Zinit'],filter_shape=fs1
                        ) - X,
                    D180,border_mode='full',
                    image_shape=config['Xshape'], filter_shape=fs2
                    )
        Z = shrinkage(Z - 1/L * gradZ,theta,k)


    def rec_error(X,Z,D):
        # Calculates the reconstruction rec_i = sum_j Z_ij * D_j
        # and the corresponding (mean) square reconstruction error
        # rec_error = (X_i - rec_i) ** 2
        rec = conv(Z,D,border_mode='valid',
                    image_shape=config['Zinit'],filter_shape=_D.shape) 
        rec_error = 0.5*T.mean(((X-rec)**2).sum(axis=-1).sum(axis=-1))
        return rec_error, rec


    sparsity = T.mean(T.sum(T.sum(T.abs_(kSparse(Z,k,1) - Z),axis=-1),axis=-1))
    re, rec = rec_error(X,Z,D) 

    return X, params, Z, rec, re, sparsity

def trainFE(layers,n_filters,filter_size,k,L,epochs,btsz,lmbd):
    data, dtrgts, valid, vtrgts, test, ttrgts, dm =  getMNIST(square=True)
    ####################
    # SET LEARNING PARAMETERS
    ####################  
    lr = .1
    momentum = 0.9
    decay = 0.95
    batches = data.shape[0]/btsz
    tbatches = test.shape[0]/btsz
    vbatches = valid.shape[0]/btsz
    print "LConvISTA -- Learned Convolutional ISTA"
    print "Epochs", epochs
    print "Batches per epoch", batches, "of size", btsz
    print

    ####################
    # INITIALIZE ALGORITHM PARAMETERS
    #################### 
    print('%i filters of size %i x %i, %i layers\nkSparse with k=%i, sparsity weighted with lambda=%f' %
          (n_filters, filter_size, filter_size, layers,k,lmbd))
    
    Dinit = {"shape": (1,n_filters,filter_size,filter_size),
             "variant": "normal", "std": 0.5}
    D = crino.initweight(**Dinit)#np.random.randn(n_filters*s,s)#
    # normalize atoms (SQUARE patches) to 1
    D /= np.sqrt((D**2).sum(axis=-2,keepdims=True).sum(axis=-1,keepdims=True))

    D180 = D[:,:,::-1,::-1].swapaxes(0,1)#D[:,:,::-1,::-1].dimshuffle(1,0,2,3)
    L = L
    theta = lmbd / L



    zinit_shape = (btsz,n_filters,
                   data[0,0].shape[0]+filter_size-1,
                   data[0,0].shape[1]+filter_size-1)
    Xshape = data[:btsz].shape


    config = {"D": D, "D180": D180, "theta": theta, "L": L, "k": k, "Zinit": zinit_shape,
              "layers": layers, "Xshape": Xshape, "btsz": btsz}
    
    # normalize weights according to this config
    norm_dic = {"D": {"axis":1, "c": 1.}}
    # threshold theta should be at least some value
    thresh_dic = {"theta": {"thresh": 1e-15}}



    ####################
    # BUILDING GRAPH
    #################### 
    X, params, Z, rec, re, sparsity = lconvista(config)
    cost = re + lmbd * sparsity
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


                print(' epoch %i, minibatch %i/%i, validation set reconstruction error %f, sparsity %f, total error %f' % (epoch, mbi + 1, batches, this_validation_loss[1],this_validation_loss[2],this_validation_loss[0]))


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

                    print('  epoch %i, minibatch %i/%i, validation set reconstruction error %f, sparsity %f, total error %f' % (epoch, mbi + 1, batches, test_score[1],test_score[2],test_score[0]))

            if patience <= iter:
                    done_looping = True
                    break
        end = time.clock()
        hours = int((end-start) / 3600)
        mins = int((end-start) / 60) - hours * 60
        secs = end - start - hours *3600 - mins *60
        print "Elapsed time for last epoch: %ih, %im, %fs" % (hours,mins,secs)
        name = 'epochs%i_layers%i_k%i_filters%i_shape%i_lambda%f_L%i' % (epochs,layers,k,n_filters,filter_size,lmbd,L/1)
        directory = "experiments/"+name
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = "sparse_epoch%i.pkl.gz" % (epoch)
        file = gzip.open("experiments/"+name+"/"+filename,'wb')
        cPickle.dump(best_params + [layers,n_filters,filter_size,k], file)  
        file.close()       
            
    return best_params,Z




def classifier(Z, Zinit):
   # Z = T.tensor4('Z')
    y = T.ivector('y')

    #rng = np.random.RandomState()

    ## Construct the first convolutional pooling layer:
    ## filtering reduces the image size to (32-5+1,32-5+1)=(28,28)
    ## maxpooling reduces this further to (28/2,28/2) = (14,14)
    ## 4D output tensor is thus of shape (btsz,firstfilters,14,14)
    #firstfilters = 20
    #layer0 = LeNetConvPoolLayer(rng, Z,
    #        image_shape=Zinit,
    #        filter_shape=(firstfilters, Zinit[1], 5, 5), poolsize=(1, 1))

    ## Construct the second convolutional pooling layer
    ## filtering reduces the image size to (14-5+1,14-5+1)=(10, 10)
    ## maxpooling reduces this further to (10/2,10/2) = (5, 5)
    ## 4D output tensor is thus of shape (btsz,secondfilters,5,5)
    #secondfilters=50    
    #layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
    #        image_shape=(Zinit[0], firstfilters, 30, 30),
    #        filter_shape=(secondfilters, firstfilters, 5, 5), poolsize=(1, 1))

    ## the HiddenLayer being fully-connected, it operates on 2D matrices of
    ## shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    ## This will generate a matrix of shape (btsz, secondfilters*5*5)
    #layer2_input = layer1.output.flatten(2)

    ## construct a fully-connected sigmoidal layer
    #layer2 = HiddenLayer(rng, input=layer2_input,
    #                     n_in=50*26*26, n_out=500,
    #                     activation=T.tanh)

    ## classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    layer3 = LogisticRegression(input=T.flatten(Z,outdim=2), n_in=int(Zinit[1]*Zinit[2]*Zinit[3]), n_out=10)

    # the cost we minimize during training is the NLL of the model
    neglog = layer3.negative_log_likelihood(y)
    cerror =  layer3.errors(y)
    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params# + layer2.params + layer1.params + layer0.params

    return neglog, cerror, params, Z, y

def testClassification():
    data, dtrgts, valid, vtrgts, test, ttrgts, dm =  getMNIST(square=True)

    file = "experiments\epochs99_layers10_k1_filters16_shape7_lambda1.000000_L10\sparse_epoch28.pkl.gz"  
    f = gzip.open(file,'rb')                              
    params = cPickle.load(f)                
    f.close()
    D = np.asarray(params[0],dtype=theano.config.floatX)
    D180 = np.asarray(params[1],dtype=theano.config.floatX)
    theta = np.asarray(params[2],dtype=theano.config.floatX)
    L = np.asarray(params[3],dtype=theano.config.floatX)
    layers = params[4]
    n_filters = params[5]
    filter_size = params[6]
    k = params[7]

    btsz = 100
    zinit_shape = (btsz,n_filters,
                   data[0,0].shape[0]+filter_size-1,
                   data[0,0].shape[1]+filter_size-1)

    config = {"D": D, "D180": D180, "theta": theta, "L": L, "Zinit": zinit_shape,
              "layers": layers, "Xshape": (btsz,1,28,28), "btsz": btsz, "k":k}


    X, params, Z, rec, re, sparsity = lconvista(config)
    #extract = theano.function([X], Z, allow_input_downcast=True)

    neglog, cerror, cparams, Z, y = classifier(Z, zinit_shape)
    
 

    




    # training ...
    lr = .5
    momentum = 0.9
    decay = 0.95
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    


    grads = T.grad(neglog,cparams)
    updates = crino.adadelta(cparams, grads, settings)
    train = theano.function([X,y], (neglog,cerror),
                            updates=updates, allow_input_downcast=True)



    lr = .1
    momentum = 0.9
    decay = 0.95
    settings = {"lr": lr, "momentum": momentum, "decay": decay}


    fullgrads = T.grad(neglog,params+cparams)
    fullupdates = crino.adadelta(params+cparams, fullgrads, settings)
    trainBoth = theano.function([X,y], (neglog,cerror),
                            updates=fullupdates, allow_input_downcast=True)




    validate = theano.function([X,y], (neglog,cerror),
                               allow_input_downcast=True)



    trainBothC = lambda X, y: trainBoth(X,y)#extract(X),y)
    trainC = lambda X, y: train(X,y)#extract(X),y)
    validateC = lambda X, y: validate(X,y)#extract(X),y)
    


    classificationTraining(data, dtrgts, valid, vtrgts, test, ttrgts,
                           btsz, (trainC,trainBothC), validateC, epochs=100, patience=2500,
                           patience_inc=1.6,  improvement=0.995)



def classificationTraining(data, dtrgts, valid, vtrgts, test, ttrgts,
                           btsz, train, validate, epochs=20, patience=2500,
                           patience_inc=1.6,  improvement=0.995):
    print '... training'

    batches = data.shape[0]/btsz
    tbatches = test.shape[0]/btsz
    vbatches = valid.shape[0]/btsz


    # early-stopping parameters
    # patience: look as this many examples regardless
    # patience_inc = wait this much longer when a new best is found
    # improvement: a rel. improvement of this much is significant
    validation_frequency = min(batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    
    epoch = 0
    done_looping = False

    t = 0 # training procedure flag

    while (epoch < epochs) and (not done_looping):
        start = time.clock()
        epoch += 1
        cost = 0
        error = 0

        for mbi in xrange(batches):
            neglog, cerror = train[t](data[mbi*btsz:(mbi+1)*btsz], dtrgts[mbi*btsz:(mbi+1)*btsz])
            print cerror
            cost += neglog
            error += cerror
            
            # iteration number
            iter = (epoch - 1) * batches + mbi
            if (iter + 1) % validation_frequency == 0:
                print 'epoch', epoch
                print('training neglog %f, classification error %f' 
                      % (cost/batches,error/batches))
                # compute zero-one loss on validation set
                v_losses = [validate(valid[i*btsz:(i+1)*btsz],
                          vtrgts[i*btsz:(i+1)*btsz]) for i in xrange(vbatches)]
                v_loss = np.mean(np.asarray(v_losses),axis=0)


                print('  validation neglog %f, classification error %f'
                      % (v_loss[0], v_loss[1]))


                if v_loss[0] < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if v_loss[0] < best_validation_loss * improvement:
                        patience = max(patience, iter * patience_inc)
                    best_validation_loss = v_loss[0]
                    best_iter = iter

                    # test it on the test set
                    t_losses = [validate(test[i*btsz:(i+1)*btsz],
                          ttrgts[i*btsz:(i+1)*btsz]) for i in xrange(tbatches)]
                    t_score = np.mean(np.asarray(t_losses),axis=0)

                    print('    test neglog %f, classification error %f' 
                          % (t_score[0] , t_score[1]))

            if patience <= iter:
                    done_looping = True
                    break
        end = time.clock()
        print "Elapsed time for last epoch", end-start, "seconds"    
        if (t == 0 and error < .02 * batches): 
            t = 1
            print '----------'
            print 'Switched to fine-tuning'
            print '----------'



def getMNIST(square=False):
    mnist_f = gzip.open("mnist.pkl.gz",'rb')                              
    train_set, valid_set, test_set = cPickle.load(mnist_f)                

    dm = train_set[0].mean(axis=0)
    dtrgts = train_set[1]
    vtrgts = valid_set[1]
    ttrgts = test_set[1]

    data = (train_set[0] - dm)
    valid = (valid_set[0] - dm)
    test = (test_set[0] - dm)
    #data = train_set[0]
    #valid = valid_set[0]
    #test = test_set[0]

    if square:
        data = data.reshape(train_set[0].shape[0],1,28,28)
        valid = valid.reshape(valid_set[0].shape[0],1,28,28)
        test = test.reshape(test_set[0].shape[0],1,28,28)
        dm = dm.reshape(1,1,28,28)

    mnist_f.close()

    return data, dtrgts, valid, vtrgts, test, ttrgts, dm

if __name__ == '__main__':
    testClassification()

