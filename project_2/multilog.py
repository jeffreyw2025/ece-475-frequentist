# Jeffrey Wong | ECE-472 | Project #2
# Strech Goal #2- Multi-Class Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(W, X):
    return np.divide(1, 1 + np.exp(-W.T @ X))

# Evaluates the softmax for class j
def class_softmax(W, X, j):
    activations = W.T @ X
    transformed_activations = np.exp(activations)/np.sum(np.exp(activations))
    return transformed_activations[j]

def SGD(X_train, Y_train, W_init, learn_rate):
    w = W_init # We assume w is a matrix with rows equal to # of parameters and cols equal to # of classes, X[i,:]s are row vectors
    num_observations = len(X_train)
    num_classes = len(Y_train[0])
    losses = np.zeros((num_observations,1))
    for i in range(num_observations):
        for j in range(num_classes):
            w[:,j] += learn_rate * (Y_train[i,j] - class_softmax(w, X_train[i,:], j)) * X_train[i,:] # This is training classes "independently"
        losses[i] = compute_base_loss(X_train, Y_train, w)
    plt.figure()
    plt.plot(np.arange(num_observations), losses)
    plt.title("Unregularized SGD log Likelihood")
    plt.savefig("loss_multilog.png")
    return w

def compute_base_loss(X, Y, w):
    loss = 0
    for m in range(len(X)):
        for n in range(len(Y[0])):
            loss += Y[m,n] * np.log(class_softmax(w, X[m,:], n))
    return loss

def generate_T_matrix(Y, num_classes): # Each row of the t array is zeroes except for a 1 in the column corresponding to a class in Y
    t = np.zeros((len(Y), num_classes))
    for i in range(num_classes):
        t[:,i] = (Y == i) # Gradually add columns to the right
    t = 1*t # Convert boolean array to int
    return t

def main():
    data_predictors = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
    num_inputs = len(data_predictors)
    num_classes = 3
    dataset = np.loadtxt("iris.data", delimiter=",")
    dataset = np.reshape(dataset, newshape=(-1, num_inputs + 1))
    # Our class correspondence is 0: Iris Setosa, 1: Iris Versicolour, 2: Iris Virginica

    dataset = np.c_[np.ones((len(dataset),1)), dataset]  # Add a column of all 1s to represent bias term
    np.random.shuffle(dataset)
    testset_start = int(np.floor(0.9 * len(dataset)))

    x_train = dataset[0:testset_start, 0:num_inputs+1]
    y_train = dataset[0:testset_start, num_inputs+1]
    x_test = dataset[testset_start:, 0:num_inputs+1]
    y_test = dataset[testset_start:, num_inputs+1]

    alpha = 0.03
    w = np.random.randn(len(data_predictors) + 1, num_classes) # W becomes a matrix under multinomal classification
    print(np.shape(w))
    w = SGD(x_train, generate_T_matrix(y_train, num_classes), w, alpha)
    y_hat = np.argmax(sigmoid(x_test.T, w), axis = 1)
    baseline_guess = np.argmax([np.sum(y_test == 0), np.sum(y_test == 1), np.sum(y_test == 2)])
    baseline_accuracy = round(100.0 * np.sum(y_test == baseline_guess) / len(y_test), 4)
    regression_accuracy = round(100.0 * np.mean(y_test == y_hat), 4)

    print("Results\n")

    accuracies = {
        "Baseline": baseline_accuracy,
        "Regression": regression_accuracy,
        "Difference": regression_accuracy-baseline_accuracy,
    }
    acc_table = pd.DataFrame(data=accuracies, index = ["Accuracy (%)"])
    print("Multinomial Logistic Regression Test Data data Accuracy by Model")
    print(acc_table)

main()
