# first-neural-network-from-scratch
a tiny neural network that learns the AND logic gate

import numpy as np

# Training Data

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ])

y = np.array([[0], [0], [0], [1]])

# Initialize Random Weights

np.random.seed(1)

w1 = np.random.randn(2, 2)
b1 = np.random.randn(1, 2)

w2 = np.random.randn(2, 1)
b2 = np.random.randn(1, 1)

# Activation (Sigmoid)

def sigmoid(z):
  return 1 / ( 1 + np.exp(-z))

def sigmoid_derivative(z):
  return sigmoid(z) * (1 - sigmoid(z))

# Forward Propagation

def forward(x):
  z1 = np.dot(x, w1) + b1
  a1 = sigmoid(z1)

  z2 = np.dot(a1, w2) + b2
  a2 = sigmoid(z2)

  return z1, a1, z2, a2

# Backpropagation

lr = 0.1 # Leaning rate

for epoch in range(5000):
  # forward 
  z1, a1, z2, a2 = forward(x)

  # Output error
  dz2 =a2 - y
  dw2 = np.dot(a1.T, dz2)
  db2 = np.sum(dz2, axis=0, keepdims=True)

  # Backpro to hidden
  dz1 = np.dot(dz2, w2.T) * sigmoid_derivative(z1)
  dw1 = np.dot(x.T, dz1)
  db1 = np.sum(dz1, axis=0, keepdims=True)

  # Update weights
  w2 -= lr * dw2
  b2 -= lr * db2
  w1 -= lr * dw1
  b1 -= lr * db1

# Test the Neural Network
print("Prediction after training")
print(forward(x)[3])
