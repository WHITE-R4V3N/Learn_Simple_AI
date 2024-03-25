# Author:       Emma Gillespie
# Date:         2024-03-25
# Description:  An AI model that can make predictions about what tasks should and could be done. 
#               There is potential for it to do multiple tasks based on output.

#### IMPORTS ####

import numpy as np

#### CODE ####

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Set initial weights and biases
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_prop(self, X):
        z1 = X.dot(self.weights1) + self.bias1
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.weights2) + self.bias2
        a2 = self.sigmoid(z2)

        return a1, a2
    
    def backwards_prop(self, X, y, learning_rate):
        m = len(X)

        a1, a2 = self.forward_prop(X)

        # Calculate the error
        delta2 = (a2 - y) * self.sigmoid_derivative(a2)
        delta1 = (delta2.dot(self.weights2.T)) * self.sigmoid_derivative(a1)

        # Update weights and biases
        self.weights2 -= learning_rate * a1.T.dot(delta2) / m
        self.weights1 -= learning_rate * X.T.dot(delta1) / m
        self.bias2 -= learning_rate * delta2.sum(axis=0) / m
        self.bias1 -= learning_rate * delta1.sum(axis=0) / m

    def predict(self, X):
        _, a2 = self.forward_prop(X)
        return a2
    
# Define network Parameters
input_size = 5
hidden_size = 15
output_size = 3
learning_rate = 0.1

model = NeuralNetwork(input_size, hidden_size, output_size)

# Training data
X = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 1, 1], [0, 1, 1, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1]])

# Train the network
epochs = 1500

for epoch in range(epochs):
    model.backwards_prop(X, y, learning_rate)

# Test the network
predictions = model.predict(np.array([[0, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 0, 0, 1, 1]])) # These would be the corpus of the text
#print(f'Predicted values: {predictions.round()}')

# Define tasks
tasks = ['Write a poem', 'Write an Article', 'Translate a sentence']
task_array = np.array([1, 1, 1])

# Match each prediction to the task_array and execute that task
for pred in predictions.round():
    print(f'Tasks for prediction {pred}:')
    if pred[0] == task_array[0]:
        print(f'{tasks[0]}')
    if pred[1] == task_array[1]:
        print(f'{tasks[1]}')
    if pred[2] == task_array[2]:
        print(f'{tasks[2]}')

    print('\n') # Makes output look cleaner
    #else:
        #print(f'Input does not satisfy task list!')