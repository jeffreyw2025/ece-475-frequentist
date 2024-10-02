# Jeffrey Wong | ECE-472 | Project #2
# Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(W, X):
    return np.divide(1, 1 + np.exp(-W.T @ X))

def SGD(X_train, Y_train, W_init, learn_rate):
    w = W_init # We assume w is a column vector, X[i,:]s are row vectors
    for i in range(len(X_train)):
        w += learn_rate * (Y_train[i] - sigmoid(w, X_train[i,:].T)) * np.atleast_2d(X_train[i,:]).T # Why can't I transpose my column vectors normally, numpy?
    return w

def scatter_matrix(predictor_names, X, Y):
    positive_cases = (Y == 1)
    negative_cases = (Y == 0)
    num_predictors = len(predictor_names)
    plt.figure()
    plt.title("Heart Disease Predictor Scatterplot Matrix")
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    for i in range(num_predictors):
        for j in range(num_predictors):
            plt.subplot(num_predictors, num_predictors, num_predictors*i+j+1)
            if i == j:
                # Diagonal entries just indicate the predictor on that row/column
                plt.text(0,0.5,predictor_names[i])
            else:
                plt.scatter(X[positive_cases, i], X[positive_cases, j], s=0.007, c='tab:blue')
                plt.scatter(X[negative_cases, i], X[negative_cases, j], s=0.007, c='tab:red')
                # Hide the axes because there's not enough space for the data itself
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
    plt.savefig("scatter_matrix.png")

def main():
    data_predictors = [
        "sbp",
        "tobacco",
        "ldl",
        "adiposity",
        "famhist",  # Note that famhist is a binary (0/1) value
        "typea",
        "obesity",
        "alcohol",
        "age"
    ]
    num_inputs = len(data_predictors) + 1
    dataset = np.loadtxt("SAheart.csv", delimiter=",")
    dataset = np.reshape(dataset, newshape=(-1, num_inputs + 1))

    dataset[:, 0] = 1  # Initialize first column to all 1s to represent bias term
    # The middle 9 columns are predictors, the last column are our (binary) labels, chd
    np.random.shuffle(dataset)
    testset_start = int(np.floor(0.8 * len(dataset)))
    validation_start = int(np.floor(0.9 * len(dataset)))

    # Replicate 4.22- 9*9 grid of plots comparing risk factors and grouping by pos/neg

    # scatter_matrix(data_predictors, dataset[:, 1:num_inputs], dataset[:, num_inputs])

    # Part 1- Basic Logistic Regression
    x_train = dataset[0:testset_start, 0:num_inputs]
    y_train = dataset[0:testset_start, num_inputs]
    x_test = dataset[testset_start:validation_start, 0:num_inputs]
    y_test = dataset[testset_start:validation_start, num_inputs]

    alpha = 0.03
    beta_init = np.zeros((num_inputs, 1))
    betas = SGD(x_train, y_train, beta_init, alpha)
    print(betas)
    y_hat = sigmoid(betas, np.atleast_2d(x_test).T)
    percent_correct_noreg = 100.0 * np.mean(y_test == (y_hat > 0.5))
    print("Baseline accuracy (just guess): 50%")
    print("Percent correct on test data with no regularization: "+ str(percent_correct_noreg) + "%")
    y_hat = sigmoid(betas, np.atleast_2d(x_train).T)
    percent_correct_noreg = 100.0 * np.mean(y_train == (y_hat > 0.5))
    print("Percent correct on training data with no regularization: "+ str(percent_correct_noreg) + "%")

np.random.seed(3257095732)
main()
