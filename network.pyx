from layer import * 
from numpy import multiply, divide, dot, square, sqrt, allclose, asarray, round as npround, clip, argmax, exp
from numpy.random import random_sample
from random import randrange
from data import * 

class network():
    def __init__(self, inputs, cf="square-mean", lr=0.001):
        self.__ins__ = inputs
        self.__loss__ = 0.0 
        self.__dloss__ = 0.0 
        self.__size__ = 0 
        
        self.accuracy = 0.0
        self.__lrate__ = lr 
        self.__cf__ = cf
        self.__layers__ = []

    def add_layer(self, neurons, activation="sigmoid"):
        new_layer = layer(self.__size__, neurons, activation)
        
        if (self.__size__ == 0):
            new_layer.__set_layer__(inputs=self.__ins__)
        else:
            new_layer.__set_layer__(inputs=self.__layers__[-1].neurons)
        
        self.__size__ += 1
        self.__layers__.append(new_layer)
    
    def forward(self, X, Y):
        for L in self.__layers__:
            if L.id == 0:
                L.__forward__(X)
            else:
                L.__forward__(self.__layers__[L.id - 1].O)

        self.__loss__, self.__dloss__ = self.__layers__[-1].__return_cost__(Y, self.__cf__)
        
        #self.__loss__ = clip(-10, 10, self.__loss__)

        if (self.__cf__ == "cross-entropy"):
            self.__layers__[-1].delta = self.__dloss__
            if (argmax(self.__layers__[-1].O) == argmax(Y)):
                self.accuracy += 1.0
                
        else:
            self.__layers__[-1].delta = multiply(self.__dloss__, self.__layers__[-1].dO) 
            if (allclose(asarray(npround(self.__layers__[-1].O)), asarray(Y), equal_nan=True)):
                self.accuracy += 1.0

    def backprop_modified(self, epoch, X, Y, e=1e-8):
        """Update Output Layer First"""
        self.forward(X, Y)
        self.__layers__[-1].__updateW__(exp(-epoch) + divide(self.__lrate__, sqrt(0.1 * square(dot(self.__layers__[-1].delta.T, self.__layers__[-1].I)).sum() + e)))
        self.__layers__[-1].__updateB__(exp(-epoch) + divide(self.__lrate__, sqrt(0.1 * square(self.__layers__[-1].delta).sum() + e )))

        for j in range(self.__size__ - 2, -1, -1):
            self.__layers__[j].delta = multiply(dot(self.__layers__[j + 1].delta, self.__layers__[j + 1].W), self.__layers__[j].dO)
            self.__layers__[j].__updateW__(exp(-epoch) + divide(self.__lrate__, sqrt(0.1 * square(dot(self.__layers__[j].delta.T, self.__layers__[j].I)).sum() + e)))
            self.__layers__[j].__updateB__(exp(-epoch) + divide(self.__lrate__, sqrt(0.1 * square(self.__layers__[j].delta).sum() + e )))

    def backprop_normal(self, X, Y, e=1e-8):
        self.forward(X, Y)
        self.__layers__[-1].__updateW__(divide(self.__lrate__, sqrt(0.1 * square(dot(self.__layers__[-1].delta.T, self.__layers__[-1].I)).sum() + e)))
        self.__layers__[-1].__updateB__(divide(self.__lrate__, sqrt(0.1 * square(self.__layers__[-1].delta).sum() + e )))

        for j in range(self.__size__ - 2, -1, -1):
            self.__layers__[j].delta = multiply(dot(self.__layers__[j + 1].delta, self.__layers__[j + 1].W), self.__layers__[j].dO)
            self.__layers__[j].__updateW__(divide(self.__lrate__, sqrt(0.1 * square(dot(self.__layers__[j].delta.T, self.__layers__[j].I)).sum() + e)))
            self.__layers__[j].__updateB__(divide(self.__lrate__, sqrt(0.1 * square(self.__layers__[j].delta).sum() + e )))

    def train(self, X, Y, epochs = 1000):
        size = len(X)
        half = size // 2
        start = randrange(0, half)
        end = randrange(half, size)
        old_acc = 0.0
        up = 0.0 
        for e in range(epochs):
            for j in range(size):
                if (j >= start and j <= end):
                    if (e < 1000 and old_acc//(e + 1) < 75):
                        self.backprop_modified(e, X[j], Y[j])
                    else:
                        self.backprop_normal(X[j], Y[j])

                else:
                    self.forward(X[j], Y[j])

            self.accuracy = self.accuracy / size * 100 
            old_acc += self.accuracy

            if (e % 100 == 0):
                print("\nEpoch: {0}/{1} Loss: {2} Accuracy: {3}%".format(e, epochs, self.__loss__, round(self.accuracy, 2)))

            if (e % 20 == 0):
                up = random_sample()
                if (up > .8):
                    start = randrange(0, half)
                    end = randrange(half, size)

            self.accuracy = 0.0

    def test(self, X, Y):
        size = len(X)
        for j in range(size):
            self.forward(X[j], Y[j])
        print("\nTest Accuracy: ", self.accuracy / size * 100)