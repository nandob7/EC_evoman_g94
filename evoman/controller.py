################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import numpy as np


class Controller(object):
    def __init__(self, n_inputs, n_hidden_neurons, n_outputs=5):
        # Initialize parameters
        self.n_inputs = n_inputs
        self.n_hidden_neurons = n_hidden_neurons
        self.n_outputs = n_outputs

        # These will be set when the genome (weights) is passed
        self.w_input_hidden = None
        self.w_hidden_output = None

    def set(self, genome, n_inputs):
        """
        Set the genome (weights) for the neural network and initialize the weights.
        The genome will be a flat array of weights that we split into input-hidden and hidden-output weights.
        """
        # Calculate the number of weights for input -> hidden and hidden -> output
        input_hidden_size = (n_inputs + 1) * self.n_hidden_neurons  # +1 for the bias term
        hidden_output_size = (self.n_hidden_neurons + 1) * self.n_outputs  # +1 for the bias term

        # Split the genome into input-hidden and hidden-output weights
        w_input_hidden = genome[:input_hidden_size]
        w_hidden_output = genome[input_hidden_size:input_hidden_size + hidden_output_size]

        # Reshape the genome into weight matrices
        self.w_input_hidden = np.reshape(w_input_hidden, (self.n_hidden_neurons, n_inputs + 1))  # +1 for bias
        self.w_hidden_output = np.reshape(w_hidden_output, (self.n_outputs, self.n_hidden_neurons + 1))  # +1 for bias

    def sigmoid(self, x):
        """ Sigmoid activation function """
        return 1.0 / (1.0 + np.exp(-x))

    def control(self, inputs, cont=None):
        """
        Feedforward the inputs through the neural network to generate boolean actions.
        params: inputs is a list or array of sensor inputs from the environment
        """
        # Add bias to the input layer (append 1 to inputs)
        inputs_with_bias = np.append(inputs, 1)  # Bias term added to inputs

        # Compute activations for the hidden layer
        hidden_activations = self.sigmoid(np.dot(self.w_input_hidden, inputs_with_bias))

        # Add bias to the hidden layer (append 1 to hidden activations)
        hidden_with_bias = np.append(hidden_activations, 1)  # Bias term added to hidden activations

        # Compute output activations
        output_activations = self.sigmoid(np.dot(self.w_hidden_output, hidden_with_bias))

        # Threshold the outputs to binary values (0 or 1) to represent actions
        actions = [1 if o > 0.5 else 0 for o in output_activations]

        # The output is a list of boolean actions: [left, right, jump, shoot, release]
        return actions

    def genome_size(self):
        """
        Calculate the total size of the genome (number of weights in the network).
        This includes:
        - Weights for input -> hidden connections (including bias)
        - Weights for hidden -> output connections (including bias)
        """
        # Number of weights between input and hidden layers (including biases)
        input_hidden_size = (self.n_inputs + 1) * self.n_hidden_neurons

        # Number of weights between hidden and output layers (including biases)
        hidden_output_size = (self.n_hidden_neurons + 1) * self.n_outputs

        # Total genome size is the sum of these two
        return input_hidden_size + hidden_output_size
