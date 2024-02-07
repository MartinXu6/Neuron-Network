
import numpy as np
import pandas as pd
import math
import random
from PIL import Image
import json

# stored weights
file = open("weights.json","r")
# weight_matrix = [[[random.uniform(-0.5, 0.5) for j in range(67500)] for i in
#                   range(50)],
#                  [[random.uniform(-0.5, 0.5) for j in range(50)] for i in
#                   range(50)],
#                  [[random.uniform(-0.5, 0.5) for j in range(50)] for i in
#                   range(50)],
#                  [[random.uniform(-0.5, 0.5) for j in range(50)] for i in
#                   range(2)]
#                  ]
weight_matrix = json.load(file)


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
    dc_dyhat = -2 * (y - y_hat)
    dyhat_dz = ((1 + math.e ** (z * -1)) ** -2) * (math.e ** (z * - 1))
    dzdw = a
    return dc_dyhat * dyhat_dz * dzdw


def dcda(y, y_hat, z, w):
    dc_dyhat = -2 * (y - y_hat)
    dyhat_dz = ((1 + math.e ** (z * -1)) ** -2) * (math.e ** (z * - 1))
    dzda = w
    return dc_dyhat * dyhat_dz * dzda


def back_propagation(current_layer, prev_layer, true_value, predictions, index):
    new_a = []
    for curr_neuron in range(len(current_layer)):
        for prev_neuron in range(len(prev_layer)):
            gradient = dcdw(true_value[curr_neuron], predictions[curr_neuron], current_layer[curr_neuron].z,
                            prev_layer[prev_neuron].activation)
            if gradient >= 0.0001:
                weight_matrix[index][curr_neuron][prev_neuron] -= 0.01
            elif gradient <= -0.0001:
                weight_matrix[index][curr_neuron][prev_neuron] += 0.01
            a_gradient = dcda(true_value[curr_neuron], predictions[curr_neuron], current_layer[curr_neuron].z,
                              weight_matrix[index][curr_neuron][prev_neuron])
            if a_gradient >= 0.0001:
                new_a.append(prev_layer[prev_neuron].activation - 0.01)
            elif a_gradient <= -0.0001:
                new_a.append(prev_layer[prev_neuron].activation + 0.01)
            else:
                new_a.append(prev_layer[prev_neuron].activation)
    return new_a


def discarding_back_propagation(current_layer, prev_layer, true_value, predictions, index):
    for curr_neuron in range(len(current_layer)):
        for prev_neuron in range(len(prev_layer)):
            gradient = dcdw(true_value[curr_neuron], predictions[curr_neuron], current_layer[curr_neuron].z,
                            prev_layer[prev_neuron])
            if gradient >= 0.0001:
                weight_matrix[index][curr_neuron][prev_neuron] -= 0.01
            elif gradient <= -0.0001:
                weight_matrix[index][curr_neuron][prev_neuron] += 0.01


# extracting Image RGB values
for num in range(1,101):
    if num % 2 == 0:
        img = Image.open(f"catsndogs/dataset/training_set/dogs/dog.{num}.jpg")
        label = [0,1]
    else:
        img = Image.open(f"catsndogs/dataset/training_set/cats/cat.{num}.jpg")
        label = [1,0]
    img = img.resize((150, 150))
    pixels = img.load()
    # extracting pixels into tuples of RGB values, and separate each pixel into three input neurons in the neuron
    # network
    input_pixels = []
    for row in range(img.size[0]):
        for col in range(img.size[1]):
            input_pixels.append(pixels[row, col])

    input_pixels = [i / 1000 for tup in input_pixels for i in tup]
    # network initialisation
    Network = Neural_Network(input_pixels)
    overall_prediction = Network.out
    # back_propagation

    cost1 = back_propagation(Network.Hidden3.neurons, Network.Hidden2.neurons, label, overall_prediction, 3)
    cost2 = back_propagation(Network.Hidden2.neurons, Network.Hidden1.neurons, cost1, Network.Hidden2.get_output(), 2)
    cost3 = back_propagation(Network.Hidden1.neurons, Network.input_layer.neurons, cost2, Network.Hidden1.get_output(),
                             1)
    discarding_back_propagation(Network.input_layer.neurons, input_pixels, cost3, Network.input_layer.get_output(), 0)
    Network = Neural_Network(input_pixels)
    print(f"Network_output{num}:", Network.out)
    print("cost=", sum([(label[i] - Network.out[i]) ** 2 for i in range(2)]) / 2,"label = ",label)
    file = open("weights.json","w")
    file.truncate()
    json.dump(weight_matrix,file)