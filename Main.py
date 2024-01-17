import numpy as np
import pandas as pd
import math
import random


class Neuron:
    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias

    def ReLU(self):
        num = np.dot(self.inputs, self.weights) + self.bias
        return max(0, num)

    def forward_pass(self):
        return self.ReLU()


class Neuron_layer:
    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.weights = weights

    def Softmax(self, outputs):
        divider = sum([math.e ** i for i in outputs])
        return [math.e ** i / divider for i in outputs]

    def get_output(self):
        neurons = [Neuron(self.inputs, self.weights, random.random()).forward_pass() for i in range(len(self.inputs))]
        return neurons

    def get_prediction(self):
        neurons = [Neuron(self.inputs, self.weights, random.random()).forward_pass() for i in range(2)]
        return neurons


Input_layer = Neuron_layer([1.3, 5.1, 2.2, 0.7, 1.1], [random.random() for i in range(5)])
Hidden1 = Neuron_layer(Input_layer.get_output(), [random.random() for k in range(5)])
Hidden2 = Neuron_layer(Hidden1.get_output(), [random.random() for x in range(5)])
Hidden3 = Neuron_layer(Hidden2.get_output(), [random.random() for j in range(5)])
Output_layer = Neuron_layer(Hidden3.get_output(), [random.random() for h in range(5)])
prediction = Output_layer.Softmax(Output_layer.get_prediction())
print(prediction)
