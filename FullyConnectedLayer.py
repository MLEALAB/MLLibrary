import numpy as np

class FullyConnectedLayer(object):
    
    def __init__(self,num_inputs,layer_size,activation_fn):
        self.W=np.random.standard_normal((num_inputs,layer_size))
        self.b=np.random.standard_normal(layer_size)
        self.size=layer_size
        self.activation=activation_fn

    def foward(self,x):
        z=np.dot(x,self.W)+self.b
        return self.activation(z)
