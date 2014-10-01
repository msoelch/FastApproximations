import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d as conv
import time
import crino
from convolutional_mlp import HiddenLayer, LeNetConvPoolLayer, LogisticRegression
from ksparse import kSparse

import gzip, cPickle, sys, os


def sh(x, theta,k,axes):
    """Simple shrinkage function.
    """
    return T.sgn(x) * T.maximum(0, T.abs_(x) - theta)

def sh_ksparse(x, theta,k,axes):
    """Advanced shrinkage function shrinking all but the k largest entries
    """
    ksZ = kSparse(x,k,axes)
    #ksZ = T.sgn(x)*kSparse(T.abs_(x),k,axes)
    return sh(x,theta,k,axes) - sh(ksZ,theta,k,axes) + ksZ



def lconvista(config):
    """Learned Convolutional ISTA   

    Returns TODO
    """
    print "[LConvISTA]"

    layers = config['layers']

    k = config['k']
    axes = config['axes']
      
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
        
    #shape information for speed up of convolution
    Dshape = _D.shape
    D180shape = _D180.shape
    Zshape = config['Zshape']
    Xshape = config['Xshape']


    #need tensor4:s due to interface of nnet.conv.conv2d
    #X.shape = (btsz, singleton dimension, h, w)
    X = T.tensor4('X', dtype=theano.config.floatX)
    #Z.shape = (btsz, n_filters, h+s-1, w+s-1)
    Z = T.zeros(Zshape,dtype=theano.config.floatX)

    shrinkage = sh_ksparse

    for i in range(layers):    
        gradZ = conv(
                    conv(
                            Z,D,border_mode='valid',
                            image_shape=Zshape,filter_shape=Dshape
                        ) - X,
                    D180,border_mode='full',
                    image_shape=Xshape, filter_shape=D180shape
                    )
        Z = shrinkage(Z - 1/L * gradZ,theta,k,axes)



    # Calculates the reconstruction rec_i = sum_j Z_ij * D_j
    # and the corresponding (mean) square reconstruction error
    # rec_error = (X_i - rec_i) ** 2
    rec = conv(Z,D,border_mode='valid',
                image_shape=Zshape,filter_shape=Dshape) 
    rec_error = 0.5*T.mean(((X-rec)**2).sum(axis=-1).sum(axis=-1))

    sparsity = T.mean(T.sum(T.sum(T.abs_(kSparse(Z,k,axes) - Z),axis=-1),axis=-1))

    return X, params, Z, rec, rec_error, sparsity



def trainFE(layers,n_filters,filter_size,k,axes,L,epochs,btsz,lmbd,note=None):
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
    D = crino.initweight(**Dinit)
    # normalize atoms (SQUARE patches) to 1
    D /= np.sqrt((D**2).sum(axis=-2,keepdims=True).sum(axis=-1,keepdims=True))

    D180 = D[:,:,::-1,::-1].swapaxes(0,1)
    L = L
    theta = lmbd / L



    Zshape = (btsz,n_filters,
                   data[0,0].shape[0]+filter_size-1,
                   data[0,0].shape[1]+filter_size-1)
    Xshape = data[:btsz].shape


    config = {"D": D, "D180": D180, "theta": theta, "L": L, "k": k, "axes": axes, "Zshape": Zshape,
              "layers": layers, "Xshape": Xshape, "btsz": btsz}
    

    
    ####################
    # BUILDING GRAPH
    #################### 
    X, params, Z, rec, re, sparsity = lconvista(config)
    cost = re + lmbd * sparsity
    grads = T.grad(cost,params)

    # training ...
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    updates = crino.adadelta(params, grads, settings)#, **norm_dic)
    
    # ... normalize D and D180
    updates[params[0]] = updates[params[0]] / T.sqrt(theano.map(lambda patch: (patch**2).sum(keepdims=True),updates[params[0]][0])[0])
    updates[params[1]] = updates[params[1]] / T.sqrt(theano.map(lambda patch: (patch**2).sum(keepdims=True),updates[params[1]][0])[0])

    train = theano.function([X], (cost, Z, re, sparsity) + tuple(params), updates=updates, allow_input_downcast=True)
    validate = theano.function([X], (cost, re, sparsity), allow_input_downcast=True)
    

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
    valfreq = min(batches, patience / 2)
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
            reprint_inline('current mbi: ' + str(mbi))
            cost, Z, re, sparsity, params[0], params[1], params[2], params[3] = train(data[mbi*btsz:(mbi+1)*btsz])

            
            # iteration number
            iter = (epoch - 1) * batches + mbi
            if (iter + 1) % valfreq == 0:
                print
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
        saveparams(epoch,epochs,layers,k,axes,n_filters,filter_size,lmbd,L/1,note,best_params)
        printtime(end-start)

    return best_params,Z






def classifier(Z, Zshape):
    y = T.ivector('y')
    lr = LogisticRegression(input=T.flatten(Z,outdim=2), n_in=int(Zshape[1]*Zshape[2]*Zshape[3]), n_out=10)
    neglog = lr.negative_log_likelihood(y)
    cerror =  lr.errors(y)
    params = lr.params
    return neglog, cerror, params, Z, y





def testClassification():
    data, dtrgts, valid, vtrgts, test, ttrgts, dm =  getMNIST(square=True)

    params = unpickle("experiments\epochs99_layers10_k1_filters16_shape7_lambda1.000000_L10_normalizationdeactivated_nomean\sparse_epoch37.pkl.gz")


    n_filters = params[5]
    filter_size = params[6]

    btsz = 100
    Zshape = (btsz,n_filters,
                   data[0,0].shape[0]+filter_size-1,
                   data[0,0].shape[1]+filter_size-1)

    config = {"D": np.asarray(params[0],dtype=theano.config.floatX),
              "D180": np.asarray(params[1],dtype=theano.config.floatX), 
              "theta": np.asarray(params[2],dtype=theano.config.floatX), 
              "L": np.asarray(params[3],dtype=theano.config.floatX), 
              "Zshape": Zshape,
              "layers": params[4], 
              "Xshape": (btsz,1,28,28), 
              "btsz": btsz, 
              "k": params[7],
              "axes": params[8]}


    X, params, Z, rec, re, sparsity = lconvista(config)

    neglog, cerror, cparams, Z, y = classifier(Z, Zshape)
    

    # training ...
    lr = 1.
    momentum = 0.9
    decay = 0.95
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    grads = T.grad(neglog,cparams)
    updates = crino.adadelta(cparams, grads, settings)
    train = theano.function([X,y], (neglog,cerror),
                            updates=updates, allow_input_downcast=True)
    trainC = lambda X, y: train(X,y)


    lr = .5
    settings = {"lr": lr, "momentum": momentum, "decay": decay}
    fullgrads = T.grad(neglog,params+cparams)
    fullupdates = crino.adadelta(params+cparams, fullgrads, settings)
    trainBoth = theano.function([X,y], (neglog,cerror),
                            updates=fullupdates, allow_input_downcast=True)
    trainBothC = lambda X, y: trainBoth(X,y)
    

    validate = theano.function([X,y], (neglog,cerror),
                               allow_input_downcast=True)
    validateC = lambda X, y: validate(X,y)
    

    #no fine tuning
    classificationTraining(data, dtrgts, valid, vtrgts, test, ttrgts,
                           btsz, (trainC,trainC), validateC, epochs=100, patience=2500,
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
    # patience_inc: wait this much longer when a new best is found
    # improvement: a rel. improvement of this much is significant
    # valfreq: go through this many minibatches before checking the network on the validation set
    valfreq = min(batches, patience / 2)
                               
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
            reprint_inline('current mbi: ' + str(mbi))
            neglog, cerror = train[t](data[mbi*btsz:(mbi+1)*btsz], dtrgts[mbi*btsz:(mbi+1)*btsz])
            cost += neglog
            error += cerror
            
            # iteration number
            iter = (epoch - 1) * batches + mbi
            if (iter + 1) % valfreq == 0:
                print
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
        printtime(end-start)
        if (t == 0 and v_loss[1] < .02): 
            t = 1
            print '----------\nSwitched to fine-tuning\n----------'








def reprint_inline(text):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(str(text))

def unpickle(file):
    f = gzip.open(file,'rb')                              
    content = cPickle.load(f)                
    f.close()
    return content

def saveparams(epoch,epochs,layers,k,axes,n_filters,filter_size,lmbd,L,note,best_params):
    L = L/1
    name = 'epochs%i_layers%i_k%i_filters%i_shape%i_lambda%f_L%i' % (epochs,layers,k,n_filters,filter_size,lmbd,L/1)
    if note != None: name += note
    directory = "experiments/"+name
    if not os.path.exists(directory): os.makedirs(directory)
    filename = "sparse_epoch%i.pkl.gz" % (epoch)
    file = gzip.open("experiments/"+name+"/"+filename,'wb')
    cPickle.dump(best_params + [layers,n_filters,filter_size,k,axes], file)  
    file.close()  

def getMNIST(square=False):
    train_set, valid_set, test_set = unpickle("mnist.pkl.gz")

    dm = train_set[0].mean(axis=0)
    dtrgts = train_set[1]
    vtrgts = valid_set[1]
    ttrgts = test_set[1]

    #data = (train_set[0] - dm)
    #valid = (valid_set[0] - dm)
    #test = (test_set[0] - dm)
    data = train_set[0]
    valid = valid_set[0]
    test = test_set[0]

    if square:
        data = data.reshape(train_set[0].shape[0],1,28,28)
        valid = valid.reshape(valid_set[0].shape[0],1,28,28)
        test = test.reshape(test_set[0].shape[0],1,28,28)
        dm = dm.reshape(1,1,28,28)


    return data, dtrgts, valid, vtrgts, test, ttrgts, dm

def printtime(t):
    hours = int(t/3600)
    mins = int(t/60) - hours*60
    secs = t - hours*3600 - mins*60
    print "Elapsed time for last epoch: %ih, %im, %fs" % (hours,mins,secs)



if __name__ == '__main__':
    testClassification()

