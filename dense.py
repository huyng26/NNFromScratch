from baselayer import Layer
import numpy as np
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)
    def forward(self, input):
        return np.dot(self.weights, input) + self.bias
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate*weights_gradient
        self.bias -= learning_rate*output_gradient
        return np.dot(self.weights.T, output_gradient)
    