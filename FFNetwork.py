import numpy as np
from FFLayer import FFLayer

class FFNetwork:
    """
    FFNetwork is a class that implements the Forward-Forward Neuron Network

    This impl is primarily designed for classification tasks.
    The network is composed of a series of FFLayers.
    It supports layer-by-layer training and trains on overall "goodness".
    Epoch is defaultly set to 10.
    """
    def __init__(self, layer_sizes: list):
        # for example, for MNIST data set, the input dim is 784(for pic) + 10(for one-hot) = 794
        self.layers = []
        self.layer_sizes = layer_sizes
        for i in range(len(layer_sizes)-1):
            self.layers.append(
                FFLayer(input_dim=layer_sizes[i], output_dim=layer_sizes[i+1])
                )
        
        # for each layer trained, layer_round plus one
        # this is a logic timer for the network trainning
        self.layer_round = 0
        
    def construct_samples(self, x: np.array, y: np.int32) -> tuple:
        # x as the sample, and y as the label
        positive = np.zeros((self.layer_sizes[0],))
        negative = np.zeros((self.layer_sizes[0],))

        # use the output dim as the one_hot size
        one_hot_size = self.layer_sizes[-1]
        right_label = np.zeros((one_hot_size,))
        right_label[y] = 1
        
        positive[:one_hot_size] = right_label
        positive[one_hot_size:] = x

        wrong_y = np.random.choice([i for i in range(one_hot_size) if i != y])
        wrong_label = np.zeros((one_hot_size))
        wrong_label[wrong_y] = 1

        negative[:one_hot_size] = wrong_label
        negative[one_hot_size:] = x

        return (positive, negative)

    def train_layer(self, layer_idx, positive_data, negative_data):
        a_p = self.layers[layer_idx].forward(positive_data)
        a_n = self.layers[layer_idx].forward(negative_data)

        g_p = self.layers[layer_idx].compute_goodness(a_p)
        g_n = self.layers[layer_idx].compute_goodness(a_n)

        self.layers[layer_idx].update_weights(positive_data, a_p, True)
        self.layers[layer_idx].update_weights(negative_data, a_n, False)
    
    def train(self, X_train, y_train, epochs=10):
        for epoch in range(epochs):
            for i in range(len(X_train)):
                x, y = X_train[i], y_train[i]

                # construct samples
                pos_sample, neg_sample = self.construct_samples(x, y)

                # train layer-by-layer
                pos_input, neg_input = pos_sample, neg_sample
                for layer_idx in range(len(self.layers)):
                    self.train_layer(layer_idx, pos_input, neg_input)
                    self.layer_round += 1

                    pos_input = self.layers[layer_idx].forward(pos_input)
                    neg_input = self.layers[layer_idx].forward(neg_input)
    
    def softmax(self, x) -> np.array:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def predict_probabilities(self, x) -> np.array:
        num_classes = self.layer_sizes[-1]
        goodness_scores = np.zeros(num_classes)

        for label in range(num_classes):
            # construct testing samples: one-hot + image
            test_sample = np.zeros(self.layer_sizes[0])

            # construct one_hot
            one_hot = np.zeros(num_classes)
            one_hot[label] = 1

            # combine one_hot + image
            test_sample[:num_classes] = one_hot
            test_sample[num_classes:] = x

            # compute the total goodness of the entire network
            current_input = test_sample
            total_goodness = 0

            for layer in self.layers:
                activations = layer.forward(current_input)
                goodness = layer.compute_goodness(activations)
                total_goodness += goodness
                current_input = activations
            
            goodness_scores[label] = total_goodness

        probabilities = self.softmax(goodness_scores)
        return probabilities
    
    def predict(self, x) -> np.int32:
        probabilities = self.predict_probabilities(x)
        return np.argmax(probabilities)
