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
        # compute activation
        a_p = self.layers[layer_idx].forward(positive_data)
        a_n = self.layers[layer_idx].forward(negative_data)

        # compute positive and negative goodness for the layer
        g_p = self.layers[layer_idx].compute_goodness(a_p)
        g_n = self.layers[layer_idx].compute_goodness(a_n)

        # update weights
        self.layers[layer_idx].update_weights(positive_data, a_p, True)
        self.layers[layer_idx].update_weights(negative_data, a_n, False)

        return (g_p, g_n)
    

    def train(self, X_train, y_train, X_test, y_test, epochs=10):
        """
        train the entire network

        takes X_trian, y_train, X_test, y_test as inputs
        epoch default to 10 as FF generally use a smaller epoch
        """
        # reset the logic time counter
        self.layer_round = 0

        for epoch in range(epochs):
            epoch_stats = {
                'layer_pos_goodness': [[] for _ in self.layers],
                'layer_neg_goodness': [[] for _ in self.layers],
                'total_goodness': []
            }
            
            # train by each sample
            for i in range(len(X_train)):
                x, y = X_train[i], y_train[i]

                # construct samples
                pos_sample, neg_sample = self.construct_samples(x, y)
                pos_input, neg_input = pos_sample, neg_sample
                total_goodness_diff = 0
                # train layer-by-layer
                for layer_idx in range(len(self.layers)):
                    (layer_pos_goodness, 
                    layer_neg_goodness
                        ) = self.train_layer(layer_idx, pos_input, neg_input)

                    # record p_g and n_g of the layer in epoch_stats
                    epoch_stats["layer_pos_goodness"][layer_idx].append(layer_pos_goodness)
                    epoch_stats["layer_neg_goodness"][layer_idx].append(layer_neg_goodness)

                    total_goodness_diff += (layer_pos_goodness - layer_neg_goodness)

                    # for each layer trained, plus one to the logic timer
                    self.layer_round += 1

                    # forward compute the output which will be the input of the next layer
                    pos_input = self.layers[layer_idx].forward(pos_input)
                    neg_input = self.layers[layer_idx].forward(neg_input)

                epoch_stats['total_goodness'].append(total_goodness_diff)

            # print the epoch stats
            self._print_epoch_stats(epoch, epoch_stats)
            # compute and print the accuracy
            test_acc = self.evaluate(X_test, y_test)
            print(f"Test accuracy: {test_acc:.4f}")
            print(f"Logic timer for train time for all layers: {self.layer_round}")
            print("---------------------------------------------------------------")
    

    def evaluate(self, X_test, y_test):
        """
        compute the accuracy on the test data set
        """
        correct = 0
        for i in range(len(X_test)):
            pred = self.predict(X_test[i])
            if pred == y_test[i]:
                correct += 1
        return correct / len(X_test)
    

    def _print_epoch_stats(self, epoch, epoch_stats):
        """
        print epoch_stats
        """
        print(f"Epoch {epoch + 1}:")
        for layer_idx in range(len(self.layers)):
            avg_pos = np.mean(epoch_stats['layer_pos_goodness'][layer_idx])
            avg_neg = np.mean(epoch_stats['layer_neg_goodness'][layer_idx])
            gap = avg_pos - avg_neg
            print(f"  Layer {layer_idx}: Pos={avg_pos:.4f}, Neg={avg_neg:.4f}, Gap={gap:.4f}")

        avg_total = np.mean(epoch_stats['total_goodness'])
        print(f"  Average total goodness difference: {avg_total:.4f}")


    def _softmax(self, x) -> np.array:
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

        probabilities = self._softmax(goodness_scores)
        return probabilities
    

    def predict(self, x) -> np.int32:
        probabilities = self.predict_probabilities(x)
        return np.argmax(probabilities)
