import numpy as np
import pandas as pd
import math
import random
from PIL import Image

# extracting Image RGB values
image = Image.open(r"D:\pycharm projects\Neuron-Network\images\cat1.jpg")
label = 0
image.thumbnail((200, 200))
pixels = image.load()


class Neuron:
    def __init__(self, connected, weights):
        self.connected = connected
        self.weights = weights
        self.bias = random.uniform(-0.5, 0.5)

    def ReLU(self, value):
        return max(0, value)

    def forward_pass(self):
        num = np.dot(self.connected, self.weights) + self.bias
        return self.ReLU(num)


class Neuron_layer:
    def __init__(self, input_neurons, neuron_num):
        self.input_neurons = input_neurons
        self.neuron_num = neuron_num
        self.weights = [[random.uniform(-0.5, 0.5) for j in range(len(self.input_neurons))] for i in
                        range(self.neuron_num)]
        self.neurons = [Neuron(self.input_neurons, self.weights[i]) for i in range(self.neuron_num)]

    def Softmax(self, outputs):
        divider = sum([math.e ** i for i in outputs])
        return [math.e ** i / divider for i in outputs]

    def dSigmoid(self):
        return

    def get_cost(self):
        return

    def get_dcost(self):
        return

    def get_output(self):
        output_neurons = [neuron.forward_pass() for neuron in self.neurons]
        return output_neurons

    def get_prediction(self):
        output_neurons = [neuron.forward_pass() for neuron in self.neurons]
        return self.Softmax(output_neurons)


# extracting pixels into tuples of RGB values, and separate each pixel into three input neurons in the neuron network

input_pixels = []
for row in range(image.size[0]):
    for col in range(image.size[1]):
        input_pixels.append(pixels[row, col])

input_pixels = [i / 1000 for tup in input_pixels for i in tup]
# input_neurons = len(input_pixels)
# network initialisation
input_layer = Neuron_layer(input_pixels, 50)
Hidden1 = Neuron_layer(input_layer.get_output(), 50)
Hidden2 = Neuron_layer(Hidden1.get_output(), 50)
Hidden3 = Neuron_layer(Hidden2.get_output(), 2)
predictions = Hidden3.get_prediction()
print(predictions)
# training
one_hot = [0, 0]
one_hot[label] = 1
cost = - np.sum(np.log(predictions) * one_hot)
print(cost)
