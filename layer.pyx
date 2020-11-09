from numpy.random import rand 
from numpy import dot, multiply, divide, ones, log, square
from activation import * 

class layer():
    def __init__(self, id, nodes, activation):
        self.id = id 
        self.W = None 
        self.I = None 
        self.O = None
        self.dO = None  
        self.B = None 
        self.delta = None
        self.act = activation
        self.neurons = nodes 

        self.__activate__ = {"sigmoid" : sigmoid, "tanh": ztanh, "relu": relu, 
        "softmax": softmax, "teq" : teq}

        self.__cost__ = {"cross-entropy" : self.cross_entropy_error, "square-mean" : self.square_mean_error, "kb" : self.kb_leibler}

    def cross_entropy_error(self, Y):
        dloss = self.O - Y 
        return multiply(-Y, log(self.O)).sum(), dloss 

    def square_mean_error(self, Y):
        dloss = self.O - Y
        return 0.5 * square(dloss).sum(), dloss

    def kb_leibler(self, Y):
        dloss = 1 + log(divide(self.O, Y))
        return multiply(self.O, log(divide(self.O, Y))).sum(), dloss

    def __set_layer__(self, inputs):
        self.W = rand(self.neurons, inputs) * inputs
        self.B = ones((1, self.neurons))
    
    def __forward__(self, X):
        self.I = X 
        self.O = dot(self.I, self.W.T) + self.B 
        self.O, self.dO = self.__activate__[self.act](self.O)

    def __return_cost__(self, Y, cf="square-mean"):
        return self.__cost__[cf](Y)
    
    def __updateW__(self, lr):
        self.W = self.W - lr * dot(self.delta.T, self.I)

    def __updateB__(self, lr):
        self.B = self.B - lr * self.delta 