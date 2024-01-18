import numpy as np
import pandas as pd
import math
import random
from PIL import Image

image = Image.open(r"D:\pycharm projects\Neuron-Network\images\cat1.jpg")
pixels = image.load()


class Neuron:
    def __init__(self, value, bias):
        self.value = value
        self.bias = bias

    def forward_pass(self):
        return self.value + self.bias


class Neuron_layer:
    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.weights = weights

    def Softmax(self, outputs):
        divider = sum([math.e ** i for i in outputs])
        return [math.e ** i / divider for i in outputs]

    def ReLU(self):
        num = np.dot(self.inputs, self.weights)
        return max(0, num)

    def get_output(self):
        ans = self.ReLU()
        neurons = [Neuron(ans,random.uniform(-0.5,0.5)).forward_pass() for i in range(len(self.inputs))]
        return neurons

    def get_prediction(self):
        ans = self.ReLU()
        neurons = [Neuron(ans,random.uniform(-0.5,0.5)).forward_pass() for i in range(2)]
        return neurons


input_pixels = []
for row in range(640):
    for col in range(426):
        input_pixels.append(pixels[row, col])
input_pixels = [i / 100 for tup in input_pixels for i in tup]
input_neurons = len(input_pixels)

Input_layer = Neuron_layer(input_pixels, [random.uniform(-1,0.5) for i in range(input_neurons)])
Hidden1 = Neuron_layer(Input_layer.get_output(), [random.uniform(-1,0.5) for k in range(input_neurons)])
Hidden2 = Neuron_layer(Hidden1.get_output(), [random.uniform(-1,0.5) for x in range(input_neurons)])
Hidden3 = Neuron_layer(Hidden2.get_output(), [random.uniform(-1,0.5) for j in range(input_neurons)])
prediction = Hidden3.Softmax(Hidden3.get_prediction())
print(prediction)
