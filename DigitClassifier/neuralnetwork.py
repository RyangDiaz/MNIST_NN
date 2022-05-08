import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, sigmoid=True):
        # Parameters of neural network
        self.neurons = [np.zeros((num_inputs, 1)) for num_inputs in layer_sizes]
        self.weights = [np.random.normal(size=(num_outputs, num_inputs)) for num_inputs, num_outputs in
                        zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.normal(size=(num_inputs, 1)) for num_inputs in layer_sizes[1:]]

        # Values of gradient stored during backpropagation
        self.neurons_delta = [np.zeros((num_inputs, 1)) for num_inputs in layer_sizes]
        self.weights_delta = [np.zeros((num_outputs, num_inputs)) for num_inputs, num_outputs in
                              zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases_delta = [np.zeros((num_inputs, 1)) for num_inputs in layer_sizes[1:]]

        self.activation_function = sigmoid

    # Used to export weights after training
    def get_weights(self):
        return self.weights

    # Used to export biases after training
    def get_biases(self):
        return self.biases

    # Used to import weights
    def set_weights(self, weights):
        self.weights = weights

    # Used to import biases
    def set_biases(self, biases):
        self.biases = biases

    # Produces an output vector from an input vector through forward propagation
    def predict(self, input_element):
        # Set the input layer of neurons as the input data
        self.neurons[0] = input_element

        # Update each layer of neurons in succession
        for i in range(1, len(self.neurons)):
            # a^(n) = W^(n-1)*x^(n-1)+b^(n-1)
            self.neurons[i] = np.matmul(self.weights[i - 1], self.neurons[i - 1]) + self.biases[i - 1]

            # Apply activation function to each neuron in the layer
            for j in range(len(self.neurons[i])):
                self.neurons[i][j] = self.activate(self.neurons[i][j])

        # Return the output layer of neurons
        return self.neurons[-1]

    # Uses sigmoid function for neuron activation
    def activate(self, parameter):
        if self.activation_function:
            return NeuralNetwork.sigmoid_activation(parameter)
        else:
            return NeuralNetwork.relu_activation(parameter)

    def activate_derivative(self, parameter):
        if self.activation_function:
            return NeuralNetwork.sigmoid_derivative(parameter)
        else:
            return NeuralNetwork.relu_derivative(parameter)

    @staticmethod
    def sigmoid_activation(parameter):
        return 1 / (1 + np.exp(-1 * parameter))

    @staticmethod
    def sigmoid_derivative(parameter):
        return NeuralNetwork.sigmoid_activation(parameter) * (1 - NeuralNetwork.sigmoid_activation(parameter))

    @staticmethod
    def relu_activation(parameter):
        return max(0, parameter)

    @staticmethod
    def relu_derivative(parameter):
        return 0 if parameter < 0 else 1

    # Trains the model through backpropagation via gradient descent on a given set of training data
    def train(self, ex_input, ex_output):
        network_output = self.predict(ex_input)

        # Calculate and output cost function
        cost = (1 / network_output.shape[0]) * sum(
            [(output - expected) ** 2 for output, expected in zip(network_output, ex_output)])

        # Find and set gradients for output layer
        for i in range(self.neurons_delta[-1].shape[0]):
            output_layer_gradient = (2 / network_output.shape[0]) * (network_output[i] - ex_output[i])
            self.neurons_delta[-1][i] = output_layer_gradient

        # Loop through remaining layers (from front to back)
        for layer in range(len(self.biases) - 1, -1, -1):
            # Find and set gradients for weights in layer (index L-1)
            for dest_index in range(self.weights_delta[layer].shape[0]):
                for source_index in range(self.weights_delta[layer].shape[1]):
                    term1 = self.neurons[layer][source_index]  # Source node on current layer
                    term2 = self.neurons[layer + 1][dest_index] * (
                            1 - self.neurons[layer + 1][dest_index])  # Operation on destination node on next layer
                    term3 = self.neurons_delta[layer + 1][dest_index]  # Gradient of destination node computed earlier
                    weight_gradient = term1 * term2 * term3
                    self.weights_delta[layer][dest_index][source_index] = weight_gradient

            # Find and set gradients for biases in layer (index L-1)
            for index in range(self.biases_delta[layer].shape[0]):
                term1 = self.neurons[layer + 1][index] * (
                        1 - self.neurons[layer + 1][index])  # Operation on corresponding node (layer + 1)
                term2 = self.neurons_delta[layer + 1][index]  # Gradient of corresponding node computed earlier
                bias_gradient = term1 * term2
                self.biases_delta[layer][index] = bias_gradient

            # Find and set gradients for neurons in layer (index L-1)
            for index in range(self.neurons_delta[layer].shape[0]):
                neuron_gradient = sum([self.neurons_delta[layer + 1][i] * self.neurons[layer + 1][i] * (
                        1 - self.neurons[layer + 1][i]) * self.weights[layer][i][index] for i in
                                       range(self.neurons[layer + 1].shape[0])])
                self.neurons_delta[layer][index] = neuron_gradient

        # Subtract gradients from all of its respective components
        for index in range(len(self.weights)):
            self.weights[index] = self.weights[index] - self.weights_delta[index]

        for index in range(len(self.biases)):
            self.biases[index] = self.biases[index] - self.biases_delta[index]

        return cost


# Driver program for training and testing network
if __name__ == "__main__":
    with np.load('mnist.npz') as data:
        training_images = data['training_images']
        training_labels = data['training_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']

    nn1 = NeuralNetwork((784, 20, 10))

    # Loading in existing parameters for network

    with np.load('networkparams_20node.npz', allow_pickle=True) as params:
        weights = params['weights']
        biases = params['biases']

    nn1.set_weights(weights)
    nn1.set_biases(biases)

    nn2 = NeuralNetwork((784, 2, 10))

    with np.load('networkparams_2node.npz', allow_pickle=True) as params:
        weights = params['weights']
        biases = params['biases']

    nn2.set_weights(weights)
    nn2.set_biases(biases)

    current = 1
    total_train = 50000

    # print_every = 100
    # for input, output in zip(training_images[:total_train], training_labels[:total_train]):
    #     cost = nn2.train(input, output)
    #     if current % print_every == 0:
    #         print(f"Trained {current} / {total_train}", cost)
    #     current += 1

    correct = 0
    total = len(test_images)
    for input, expected in zip(test_images[:],
                               test_labels[:]):
        output = nn1.predict(input)
        if np.argmax(expected) == np.argmax(output):
            correct += 1
        # print(f"Expected: {np.argmax(expected)}, Predicted: {np.argmax(output)}")
    print(f"20-Node Total Correct: {correct}/{total} ({(correct / total) * 100})%")

    correct = 0
    for input, expected in zip(test_images[:],
                               test_labels[:]):
        output = nn2.predict(input)
        if np.argmax(expected) == np.argmax(output):
            correct += 1
        # print(f"Expected: {np.argmax(expected)}, Predicted: {np.argmax(output)}")
    print(f"2-Node Total Correct: {correct}/{total} ({(correct / total) * 100})%")

    print(nn1.predict(test_images[5010]), np.argmax(nn1.predict(test_images[5010])), "Expected:", np.argmax(test_labels[5010]))

    # Exporting trained parameters of network
    # np.savez('networkparams_2node', weights=nn2.get_weights(), biases=nn2.get_biases())
