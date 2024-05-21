# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-03-25 21:33:35
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-03-25 21:33:48


import numpy as np


def sigmoid(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def xor_neural_network(A, B):
    # Weights for the first layer
    w1 = np.array([1, 1])
    w2 = np.array([1, 1])
    b1 = -0.5
    b2 = -1.5

    # Weights for the second layer
    w3 = np.array([1, -2])
    b3 = -0.5

    # First layer (hidden layer)
    H1 = sigmoid(np.dot(w1, [A, B]) + b1)
    H2 = sigmoid(np.dot(w2, [A, B]) + b2)

    # Second layer (output layer)
    out = sigmoid(np.dot(w3, [H1, H2]) + b3)

    return out > 0.5  # Returns True if output is greater than 0.5, otherwise False


# Testing the XOR function
print(xor_neural_network(0, 0))
print(xor_neural_network(0, 1))
print(xor_neural_network(1, 0))
print(xor_neural_network(1, 1))
