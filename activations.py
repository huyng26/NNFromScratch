from activation import Activation
from baselayer import Layer
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x:  1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)

class RELU(Activation):
    def __init__(self):
        def RELU(x):
            return x if x > 0 else 0
        def RELU_prime(x):
            return 1 if x > 0 else 0
        super().__init__(RELU, RELU_prime)

class Sigmoid(Activation):
    def __init__(self, activation, activation_prime):
        def sigmoid(x):
            return 1/(1+ np.exp(-x))
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s*(1-s)
        super.__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input - np.max(input))
        self.output = tmp/np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)