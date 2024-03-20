# Author:       Emma Gillespie
# Date:         2024-03-18
# Description:  A simple neural network model to be used for an AI. Using only python3 and numpy.

#### Imports ####

import numpy as np

#### Code ####

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    # Define the activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Define the derivative of the activation function
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Forward Propagation
    def forward_prop(self, X):
        z1 = X.dot(self.weights1) + self.bias1
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.weights2) + self.bias2
        a2 = self.sigmoid(z2)

        return a1, a2
    
    # Backwards propagation
    def backwards_prop(self, X, y, learning_rate):
        m = len(X)
        a1 , a2 = self.forward_prop(X)

        # Calculate error
        delta2 = (a2 - y) * self.sigmoid_derivative(a2)
        delta1 = (delta2.dot(self.weights2.T)) * self.sigmoid_derivative(a1)

        # Update weights and biases
        self.weights2 -= learning_rate * a1.T.dot(delta2) / m
        self.weights1 -= learning_rate * X.T.dot(delta1) / m
        self.bias2 -= learning_rate * delta2.sum(axis=0) / m
        self.bias1 -= learning_rate * delta1.sum(axis=0) / m

    # Predict Output
    def predict(self, X):
        _, a2 = self.forward_prop(X)
        return a2
    
#### Example Usage ####
np.random.seed(1) # Set seed for reproducibility

# Define network
input_size = 2
hidden_size = 4
output_size = 1

# Create neural network
model = NeuralNetwork(input_size, hidden_size, output_size)

# Sample training data (XOR example)
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Training the network
learning_rate = 0.1
epochs = 1500

for epoch in range(epochs):
    model.backwards_prop(X, y, learning_rate)

# Test the network
predictions = model.predict(np.array([[0, 1], [1, 0], [0, 0]]))
print(f'Predicted XOR values: {predictions.round()}')