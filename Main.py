import numpy


class Neuron:
    def __init__(self, inputs, weights, bias, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights
        self.bias = bias
        pass

    def calculate(self):
        return numpy.dot(self.inputs, self.weights) + self.bias


class Neuron_layers:
    def __init__(self):
        pass
