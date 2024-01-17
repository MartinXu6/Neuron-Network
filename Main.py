import numpy
import math


class Neuron:
    def __init__(self, inputs, weights, bias, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights
        self.bias = bias
        pass

    def ReLu(self):
        num = numpy.dot(self.inputs, self.weights) + self.bias
        return 0 if num <= 0 else num

    def forward_pass(self):
        return self.ReLu()


class Neuron_layer:
    def __init__(self):
        pass


class out_layer(Neuron_layer):
    def Softmax(self, outputs):
        divider = sum([math.e ** i for i in outputs])
        return [math.e ** i / divider for i in outputs]


n1 = out_layer()
print(n1.Softmax([1.3, 5.1, 2.2, 0.7, 1.1]))
