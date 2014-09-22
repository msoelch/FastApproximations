import theano
import theano.tensor as T
from theano.gradient import grad_undefined
import numpy as np

# taken from Max Karl, adjusted by Max Soelch for Conv. Sparse Coding (--> 4D tensor sparse codes)

class KSparseGrad(theano.Op):
    """ ksparse gradient """
    def __eq__(self,other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "KSparseWRT"
        
    def make_node(self, I, k, dCdH):
        """
            :param I: input vector
            :param k: number of non-zero elements
        """

        I_ = T.as_tensor_variable(I)
        k_ = T.as_tensor_variable(k)
        dCdH_ = T.as_tensor_variable(dCdH)

        node = theano.Apply(self, inputs=[I_, k_, dCdH_], outputs = [ T.TensorType(I_.dtype, (I_.broadcastable[0],I_.broadcastable[1],I_.broadcastable[2],I_.broadcastable[3]))() ] )

        return node
     
    def perform(self, node, inputs, output_storage):
        I, k, dCdH = inputs
        #print "KSparse python code"

        H = np.copy(dCdH)
        s = np.argsort(-abs(I),axis=1)
        for n in range(I.shape[0]):
            for i in range(I.shape[2]):
                for j in range(I.shape[3]):
                    H[n,s[n,k:,i,j],i,j] = 0.0

        output_storage[0][0] = H

    def infer_shape(self, node, input_shapes):
        I, k, dCdH = node.inputs
        I_shape, k_shape, dCdH_shape = input_shapes

        return [ I_shape ]

class KSparse(theano.Op):
    """ ksparse funktion """
    def __eq__(self,other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "kSparse"
        
    def make_node(self, I, k):
        """
            :param I: input vector
            :param k: number of non-zero elements
        """

        I_ = T.as_tensor_variable(I)
        k_ = T.as_tensor_variable(k)

        node = theano.Apply(self, inputs=[I_, k_], outputs = [ T.TensorType(I_.dtype, (I_.broadcastable[0],I_.broadcastable[1],I_.broadcastable[2],I_.broadcastable[3]))() ] )

        return node
                                                   
    def grad(self, inputs, output_gradients):
        I,k = inputs
        dCdH ,= output_gradients

        dCdI = KSparseGrad()(I,k,dCdH)
        dCdI = T.patternbroadcast(dCdI, I.broadcastable)
        
        dCdk = grad_undefined(self,1,inputs[1],
                "The gradient of ksparse with respect to the number"+\
                " of non-zero elements is undefined")

        if 'name' in dir(dCdH) and dCdH.name is not None:
            dCdH_name = dCdH.name
        else:
            dCdH_name = 'anon_dCdH'

        if 'name' in dir(I) and I.name is not None:
            I_name = I.name
        else:
            I_name = 'anon_I'

        if 'name' in dir(k) and k.name is not None:
            k_name = k.name
        else:
            k_name = 'anon_k'

        dCdI.name = 'KSparse_dCdI(dCdH='+dCdH_name+',I='+I_name+')'
        dCdk.name = 'KSparse_dCdk(dCdH='+dCdH_name+',I='+I_name+',k='+k_name+')'

        return [ dCdI, dCdk ]
        
    def perform(self, node, inputs, output_storage):
        I, k = inputs
        #print "KSparse python code"
        
        H = np.copy(I)
        s = np.argsort(-abs(I),axis=1)
        for n in range(I.shape[0]):
            for i in range(I.shape[2]):
                for j in range(I.shape[3]):
                    H[n,s[n,k:,i,j],i,j] = 0.0
    
        output_storage[0][0] = H

    def infer_shape(self, node, input_shapes):
        I,k = node.inputs
        I_shape, k_shape = input_shapes

        return [ I_shape ]
        
global kSparse
kSparse = KSparse()  

if __name__ == "__main__":
    x = theano.tensor.tensor4()
    f = theano.function([x],[kSparse(x,1)])
    print f(np.random.random((2,3,4,5)).astype('float32'))

    f_grad = theano.function([x],[T.grad(1000.0*np.mean(kSparse(x,10)*x),x)])
    print f_grad(np.random.random((2,3,4,5)).astype('float32'))
        
        
        
