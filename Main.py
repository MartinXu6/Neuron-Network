import numpy as np
import pandas as pd
import math
import random
from PIL import Image

# extracting Image RGB values
image = Image.open(r"images\cat1.jpg")
label = [0, 1]
image.thumbnail((200, 200))
pixels = image.load()
# extracting pixels into tuples of RGB values, and separate each pixel into three input neurons in the neuron network
input_pixels = []
for row in range(image.size[0]):
    for col in range(image.size[1]):
        input_pixels.append(pixels[row, col])

input_pixels = [i / 1000 for tup in input_pixels for i in tup]
# stored weights
weight_matrix = [[[random.uniform(-0.5, 0.5) for j in range(len(input_pixels))] for i in
                  range(50)],
                 [[random.uniform(-0.5, 0.5) for j in range(50)] for i in
                  range(50)],
                 [[random.uniform(-0.5, 0.5) for j in range(50)] for i in
                  range(50)],
                 [[random.uniform(-0.5, 0.5) for j in range(50)] for i in
                  range(2)]
                 ]


class Neuron:
    def __init__(self, connected, weights):
        self.connected = connected
        self.weights = weights
        self.bias = random.uniform(-1, 1)
        self.z = np.dot(self.connected, self.weights)
        self.activation = self.Sigmoid(self.z)

    def Sigmoid(self, value):
        return 1 / (1 + math.e ** (value * -1))


class Neuron_layer:
    def __init__(self, input_neurons, neuron_num, weights):
        self.input_neurons = input_neurons
        self.neuron_num = neuron_num
        self.weights = weights
        self.neurons = [Neuron(self.input_neurons, self.weights[i]) for i in range(self.neuron_num)]

    def Softmax(self, outputs):
        divider = sum([math.e ** i for i in outputs])
        return [math.e ** i / divider for i in outputs]

    def get_output(self):
        output_neurons = [neuron.activation for neuron in self.neurons]
        return output_neurons

    def get_prediction(self):
        output_neurons = [neuron.activation for neuron in self.neurons]
        return self.Softmax(output_neurons)


class Neural_Network:
    def __init__(self, picture):
        self.picture = picture
        self.input_layer = Neuron_layer(self.picture, 50, weight_matrix[0])
        self.Hidden1 = Neuron_layer(self.input_layer.get_output(), 50, weight_matrix[1])
        self.Hidden2 = Neuron_layer(self.Hidden1.get_output(), 50, weight_matrix[2])
        self.Hidden3 = Neuron_layer(self.Hidden2.get_output(), 2, weight_matrix[3])
        self.out = self.Hidden3.get_output()


# training
def dcdw(y, y_hat, z, a):
    dc_dyhat = 2 * (y - y_hat)
    dyhat_dz = ((1 + math.e ** (z * -1)) ** -2) * (math.e ** (z - 1))
    dzdw = a
    return dc_dyhat * dyhat_dz * dzdw


def dcda(y, y_hat, z, w):
    dc_dyhat = 2 * (y - y_hat)
    dyhat_dz = ((1 + math.e ** (z * -1)) ** -2) * (math.e ** (z - 1))
    dzda = w
    return dc_dyhat * dyhat_dz * dzda


# network initialisation
Network = Neural_Network(input_pixels)
predictions = Network.out
print(predictions)
# back_propagation
cost = [(label[i] - predictions[i]) ** 2 for i in range(2)]
for curr_neuron in range(len(Network.Hidden3.neurons)):
    for prev_neuron in range(len(Network.Hidden2.neurons)):
        gradient = dcdw(label[curr_neuron], predictions[curr_neuron], Network.Hidden3.neurons[curr_neuron].z,
                        Network.Hidden2.neurons[prev_neuron].activation)
        if gradient >= 0.001:
            weight_matrix[3][curr_neuron][prev_neuron] -= 0.01
        elif gradient <= -0.001:
            weight_matrix[3][curr_neuron][prev_neuron] += 0.01
        a_gradient = dcda(label[curr_neuron], predictions[curr_neuron], Network.Hidden3.neurons[curr_neuron].z,
                          weight_matrix[3][curr_neuron][prev_neuron])

