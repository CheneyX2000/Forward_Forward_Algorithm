import numpy as np
from FFLayer import FFLayer

class FFNetwork:
    def __init__(self, layer_sizes: list):
        # for example, for MNIST data set, the input dim is 784(for pic) + 10(for one-hot) = 794
        self.layers = []
        self.layer_sizes = layer_sizes
        for i in range(len(layer_sizes)-1):
            self.layers.append(
                FFLayer(input_dim=layer_sizes[i], output_dim=layer_sizes[i+1])
                )
        
    def construct_samples(self, x: np.array, y: int) -> tuple:
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
    
    def softmax(self, x):
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
    
    def predict(self, x) -> int:
        probabilities = self.predict_probabilities(x)
        return np.argmax(probabilities)
