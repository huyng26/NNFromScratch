class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, output):
        #return output
        pass
    def backward(self, output_gradient, learning_rate):
        #update parameters and return input gradient
        pass