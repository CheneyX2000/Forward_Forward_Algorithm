This is a numpy implementation of the Forward-Forward Algorithm.
["The Forward-Forward Algorithm: A New Way of Training Neural Networks" by Geoffrey Hinton](https://arxiv.org/abs/2212.13345)

This repository is for educational and experimental research.

The FFN (Forward-Forward Neural Network) is a network that can achieve cross-layer concurrency by pipelining. This is done by separating gradients descending process of each layer.
Each layer's gradient descent is based on a "Goodness function," which is essentially a Contrastive Learning process.
The "Goodness" = Positive_activation - Negative_activation.
The goal is to independently increase the Goodness of each layer so that each layer can be trained with no dependency on preceding layers. This mechanism enables almost parallel training across layers; each layer does not have to wait for propagation like in a typical back-propagation network.
The weakness of this mechanism is that it is naturally "Greedy": each preceding layer attempts to capture patterns and leaves the uncaptured information to subsequent layers. Consequently, the FFN may more frequently converge to a suboptimal solution compared to a BP network.

In this implementation, we use the following design:
- Xavier Glorot initialization for weights.
- tanh() as the activation function.
- Layer activation is computed as the sum of the squares of each neuron's activation value (as described in the original paper by Geoffrey Hinton).

Project Structure:
- FFN/ (The numpy implementation of FFN and its layer)
- math_notes/ (The math process for FFN)
- log/ (Manually stored txt log for test and training scripts)
- train/ (The test and training scripts)
- README.md

Download the code by:
```
git clone https://github.com/CheneyX2000/Forward_Forward_Algorithm.git
```

After downloading, start with the following commands:
```
conda create --name forward_forward_env python=3.10
conda activate forward_forward_env
```

Then install necessary packages:
```
conda install numpy
conda install jupytor
conda install tensorflow
```

License:
MIT