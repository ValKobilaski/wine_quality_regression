from util import load_data
import numpy as np
import sys

class linear_regression():
    def __init__(self, a, epochs):
        self.a = a
        self.epochs = epochs
        self.weights = None
        self.bias = 0


    def fit(self, x_train, y_train):

        self.weights = np.zeros(x_train.shape[0])
        n = x_train.shape[1]
        epochs = 0

        delta_loss = float('inf')

        while delta_loss >= 0.05:
            epochs += 1
            y_pred = np.dot(x_train, self.weights)+ self.bias

            d_w = (-2/n) * np.dot(x_train.T, (y_train - y_pred))
            d_b = (-2/n) * sum(y_train - y_pred)

            self.weights = self.a * (self.weights - d_w)
            self.bias = self.a * (self.bias - d_b)

            mse = (1/n) * sum((y_train - y_pred)**2)
            delta_loss = min(delta_loss, mse)





