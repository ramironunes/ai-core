# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date: 2024-03-19 18:15:06
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-03-25 21:20:59


class McCullochPittsNeuron:

    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold
    
    def process(self, inputs):
        # Multiplies each input by its corresponding weight and checks if the sum exceeds the threshold
        return 1 if sum(x * w for x, w in zip(inputs, self.weights)) >= self.threshold else 0


# Define the weights and thresholds for each logical operation
and_weights = [1, 1]
and_threshold = 2

or_weights = [1, 1]
or_threshold = 1

xnor_weights = [-1, -1]
xnor_threshold = -1

xor_weights = [1, 1]  # A single McCulloch-Pitts neuron cannot represent XOR
xor_threshold = 0  # However, included for completeness

# Create instances of the neuron for each logical operation
and_neuron = McCullochPittsNeuron(and_weights, and_threshold)
or_neuron = McCullochPittsNeuron(or_weights, or_threshold)
xnor_neuron = McCullochPittsNeuron(xnor_weights, xnor_threshold)
# xor_neuron = McCullochPittsNeuron(xor_weights, xor_threshold)  # Not applicable

# List of inputs
list_x1 = [0, 1, 0, 1]
list_x2 = [0, 0, 1, 1]
inputs = zip(list_x1, list_x2)

# Process the logical operations for each pair of inputs
print("AND:", [and_neuron.process(x) for x in inputs])
print("OR:", [or_neuron.process(x) for x in zip(list_x1, list_x2)])
print("XNOR:", [xnor_neuron.process(x) for x in zip(list_x1, list_x2)])
# print("XOR:", [xor_neuron.process(x) for x in zip(list_x1, list_x2)])  # Will not work

# Note: XOR cannot be represented with a single McCulloch-Pitts neuron.
# A set of neurons or a network is needed to represent the XOR operation.
