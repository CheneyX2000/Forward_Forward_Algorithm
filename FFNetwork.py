import numpy as np
from FFLayer import FFLayer

class FFNetwork:
    def __init__(self, layer_sizes: list):
        # for MNIST data set, the input dim is 784(for pic) + 10(for one-hot) = 794
        self.layers = []
        self.layer_sizes = layer_sizes
        for i in range(len(layer_sizes)-1):
            self.layers.append(
                FFLayer(input_dim=layer_sizes[i], output_dim=layer_sizes[i+1])
                )
        
    def construct_samples(self, x: np.array, y: int):
        # x as the sample, and y as the label
        positive = np.zeros((self.layer_sizes[0],))
        negative = np.zeros((self.layer_sizes[0],))

        right_label = np.zeros((self.layer_sizes[-1]))
        for i in range(len(right_label)):
            positive[i] = 
        for j in range(len(y), len(x)):
            positive[j] = x[i]

        wrong_label = np.random.choice([i for i in range(len(y)) if i != y])
        
        # how to implement negative?


        return (positive, negative)

    def train_layer(self, layer_idx, positive_data, negative_data):
        a_p = self.layers[layer_idx].forward(positive_data)
        a_n = self.layers[layer_idx].forward(negative_data)

        g_p = self.layers[layer_idx].compute_goodness(a_p)
        g_n = self.layers[layer_idx].compute_goodness(a_n)

        self.layers[layer_idx].update_weights(positive_data, a_p, True)
        self.layers[layer_idx].update_weights(negative_data, a_n, False)

    def predict(self, x):
        for i in range(self.layer_sizes-1):
            x = self.layers[i].forward(x)
        return x