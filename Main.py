import numpy as np
import pandas as pd
import math
import random
from PIL import Image

# extracting Image RGB values
image = Image.open(r"D:\pycharm projects\Neuron-Network\images\cat1.jpg")
label = [0,1]
image.thumbnail((200, 200))
pixels = image.load()


class Neuron:
    def __init__(self, connected, weights):
        self.connected = connected
        self.weights = weights
        self.bias = random.uniform(-0.5, 0.5)

    def Sigmoid(self, value):
        return 1 / (1 + math.e ** (value * -1))

    def forward_pass(self):
        num = np.dot(self.connected, self.weights) + self.bias
        return self.Sigmoid(num)


class Neuron_layer:
    def __init__(self, inputs, neuron_num):
        self.inputs = inputs
        self.neuron_num = neuron_num
        self.weights = [[random.uniform(-0.3, 0.3) for j in range(len(self.inputs))] for i in range(self.neuron_num)]

    def Softmax(self, outputs):
        divider = sum([math.e ** i for i in outputs])
        return [math.e ** i / divider for i in outputs]

    def get_output(self):
        neurons = [Neuron(self.inputs, self.weights[i]).forward_pass() for i in range(self.neuron_num)]
        return neurons

    # def get_prediction(self):
    #     neurons = [Neuron(self.inputs, self.weights[i]).forward_pass() for i in range(self.neuron_num)]
    #     return self.Softmax(neurons)


# extracting pixels into tuples of RGB values, and separate each pixel into three input neurons in the neuron network

input_pixels = []
for row in range(image.size[0]):
    for col in range(image.size[1]):
        input_pixels.append(pixels[row, col])

input_pixels = [i / 1000 for tup in input_pixels for i in tup]
input_neurons = len(input_pixels)
# network initialisation
input_layer = Neuron_layer(input_pixels, 50)
Hidden1 = Neuron_layer(input_layer.get_output(), 50)
Hidden2 = Neuron_layer(Hidden1.get_output(), 50)
Hidden3 = Neuron_layer(Hidden2.get_output(), 2)
predictions = Hidden3.get_output()
print(predictions)
# training

