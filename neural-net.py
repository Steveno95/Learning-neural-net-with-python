import copy, numpy as np
np.random.seed(0)

# sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1  +np.exp(-x))
    return output

# convert output of sigmoid to its derivative
def sigmoid_to_derivative(output):
    return output * (1 - output)

