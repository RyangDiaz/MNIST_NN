import canvas as can
from neuralnetwork import NeuralNetwork
import numpy as np


def run_canvas():
    nn = NeuralNetwork((784, 64, 10))

    with np.load('canvas_network_params.npz', allow_pickle=True) as params:
        weights = params['weights']
        biases = params['biases']

    nn.set_weights(weights)
    nn.set_biases(biases)

    canvas = can.App(classifier=nn, width=280, height=280)
    canvas.root.mainloop()

    # Upon exiting program, save existing nn parameters to a file
    np.savez('canvas_network_params', weights=canvas.network.get_weights(), biases=canvas.network.get_biases())
    print("Network parameters saved to 'canvas_network_params.npz'")


if __name__ == '__main__':
    run_canvas()
