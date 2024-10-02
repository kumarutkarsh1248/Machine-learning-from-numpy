import numpy as np
import pandas as pd 
import matplotlib.pyplot as pyplot

class Linear_Regression:
    def __init__(self, X_train, y_train):
        self.n_points = X_train.shape[0]
        self.n_features = X_train.shape[1]

        self.X_train = X_train
        self.y_train = y_train

        # Initialize weights to zero to ensure proper convergence
        self.weights = np.zeros((1, self.n_features))
        self.bias = 0

        self.epoch = 1000
        self.learning_rate = 0.05

    def fit(self, epoch=1000, learning_rate=0.05):
        self.epoch = epoch
        self.learning_rate = learning_rate

        # calculate the gradient
        self.gradient_descent()

    def loss_function(self):
        total_error = self.y_train - (np.sum(self.X_train * self.weights, axis=1) + self.bias)
        return np.mean(total_error**2)

    # Gradient descent with mean square error
    def gradient_descent(self):
        for i in range(self.epoch):
            # Forward pass: predicted values
            y_pred = np.dot(self.X_train, self.weights.T) + self.bias
            
            # Error calculation
            error = self.y_train - y_pred
            
            # Calculate gradients
            m_gradient = (-2 / self.n_points) * np.dot(error.T, self.X_train)
            b_gradient = (-2 / self.n_points) * np.sum(error)
            
            # Update weights and bias
            self.weights -= self.learning_rate * m_gradient
            self.bias -= self.learning_rate * b_gradient

            # Optional: Track progress
            if i % 10000 == 0:
                loss = self.loss_function()
                print(f"Epoch {i}, Loss: {loss}")

    def predict(self, X_test):
        return np.dot(X_test, self.weights.T) + self.bias