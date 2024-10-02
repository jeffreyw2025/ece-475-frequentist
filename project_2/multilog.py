# Jeffrey Wong | ECE-472 | Project #2
# Extra Credit- Multi-Class Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(W, X):
    return np.divide(1, 1 + np.exp(-W.T @ X))

def SGD(X_train, Y_train, W_init, learn_rate):
    w = W_init # We assume w is a column vector, X[i,:]s are row vectors
    for i in range(len(X_train)):
        w += learn_rate * (Y_train[i] - sigmoid(w, X_train[i,:].T)) * np.atleast_2d(X_train[i,:]).T 
        # Why can't I transpose my column vectors normally, numpy?
    return w

def main():
    data_predictors = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
    num_inputs = len(data_predictors)
    dataset = np.loadtxt("iris.data", delimiter=",")
    dataset = np.reshape(dataset, newshape=(-1, num_inputs + 1))
    # Our class correspondence is 0: Iris Setosa, 1: Iris Versicolour, 2: Iris Virginica

    dataset = np.c_[np.ones((len(dataset),1)), dataset]  # Add a column of all 1s to represent bias term
    np.random.shuffle(dataset)
    testset_start = int(np.floor(0.8 * len(dataset)))
    validation_start = int(np.floor(0.9 * len(dataset)))

    x_train = dataset[0:testset_start, 0:num_inputs]
    y_train = dataset[0:testset_start, num_inputs]
    x_test = dataset[testset_start:validation_start, 0:num_inputs]
    y_test = dataset[testset_start:validation_start, num_inputs]

    alpha = 0.04
    beta_init = np.zeros((num_inputs, 1))

np.random.seed(3257095732)
main()
