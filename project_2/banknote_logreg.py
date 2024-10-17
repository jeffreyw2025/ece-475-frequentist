# Jeffrey Wong | ECE-472 | Project #2
# Logistic Regression- Banknote Authenticity Data
# Data found at https://archive.ics.uci.edu/dataset/267/banknote+authentication

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector as sfs

def sigmoid(W, X):
    return np.divide(1, 1 + np.exp(-W.T @ X))

# Part 1- No Regularization
def SGD_unreg(X_train, Y_train, W_init, learn_rate):
    w = W_init # We assume w is a column vector, X[i,:]s are row vectors
    num_observations = len(X_train)
    for i in range(num_observations):
        for j in range(len(X_train[0])):
            w[j] += learn_rate * (Y_train[i] - sigmoid(w, X_train[i,:])) * X_train[i,j]
    return w

# Part 3- L2 Regularization
def SGD_L2(X_train, Y_train, W_init, learn_rate, lbd):
    w = W_init
    num_observations = len(X_train)
    for i in range(num_observations):
        for j in range(len(X_train[0])):
            ridge_penalty = (0 if j == 0 else lbd * w[j]) # Don't penalize bias!
            w[j] += learn_rate * (Y_train[i] - sigmoid(w, X_train[i,:]) - ridge_penalty) * X_train[i,j]
    return w

# Strech Goal 1- L1 Regularization
def SGD_L1(X_train, Y_train, W_init, learn_rate, lbd):
    w = W_init
    u = 0 # "Theoretical" Cumulative penalty term
    q = np.zeros((1,len(w))) # "Actual" cumulative penalty term
    num_observations = len(X_train)
    for i in range(num_observations):
        u += learn_rate * lbd
        for j in range(len(X_train[0])):
            if j==0:  
                w[j] += learn_rate * (Y_train[i] - sigmoid(w, X_train[i,:])) * X_train[i,j]
            else:
                w_half = w[j] + learn_rate * (Y_train[i] - sigmoid(w, X_train[i,:])) * X_train[i,j]
                if w_half > 0:
                    w[j] = max(0, w_half - u - q[:,j-1])
                elif w_half < 0:
                    w[j] = min(0, w_half + u - q[:,j-1])
                q[:,j-1] += (w[j] - w_half)
    return w

def compute_base_loss(X, Y, w):
    loss = 0
    for k in range(len(X)):
        loss += Y[k]*np.log(sigmoid(w, X[k,:])) + (1-Y[k])*np.log(1-sigmoid(w, X[k,:]))
    return loss

def main():
    data_predictors = [
        "variance",
        "skewness",
        "curtosis",
        "entropy",
    ]
    num_inputs = len(data_predictors) + 1
    dataset = np.loadtxt("data_banknote_authentication.txt", delimiter=",")
    dataset = np.reshape(dataset, newshape=(-1, num_inputs))

    bias = np.ones((len(dataset), 1))
    dataset = np.concatenate((bias, dataset), axis=1)  # Add column of all 1s to represent bias term
    # The middle 4 columns are predictors, the last column are our (binary) labels, chd
    np.random.shuffle(dataset)
    testset_start = int(np.floor(0.8 * len(dataset)))
    validation_start = int(np.floor(0.9 * len(dataset)))

    # Normalize non-intercept predictors so that we're not passing big numbers into the sigmoid
    dataset[:, 1:num_inputs] = np.divide(
        dataset[:, 1:num_inputs] - np.mean(dataset[:, 1:num_inputs], axis=0),
        np.sqrt(np.var(dataset[:, 1:num_inputs], axis=0)),
    )

    # Part 1- Basic Logistic Regression
    x_train = dataset[0:testset_start, 0:num_inputs]
    y_train = dataset[0:testset_start, num_inputs]
    x_test = dataset[testset_start:validation_start, 0:num_inputs]
    y_test = dataset[testset_start:validation_start, num_inputs]

    alpha = 0.005
    beta_init = np.random.randn(num_inputs, 1)
    print("Training unregularized...\n")
    beta_unreg = SGD_unreg(x_train, y_train, beta_init, alpha)
    y_hat = sigmoid(beta_unreg, x_test.T)
    baseline_guess = 1 if sum(y_train == 1)/(len(y_train)) > 0.5 else 0
    baseline_accuracy = round(100.0 * np.sum(y_test == baseline_guess) / len(y_test), 4)
    noreg_accuracy = round(100.0 * np.mean(y_test == (y_hat > 0.5)), 4)

    # Part 2- Stepwise Selection
    lr = LogisticRegression(penalty = None)
    print("Training stepwise...\n")
    sfs_forward = sfs(
        lr, direction="forward", scoring='accuracy'
    ).fit(x_train, y_train)
    predictors = np.array(["bias"] + data_predictors)
    selected_predictors = predictors[sfs_forward.get_support()]
    beta_stepwise = np.multiply(beta_unreg.T, sfs_forward.get_support()).T
    y_hat = sigmoid(beta_stepwise, x_test.T)
    stepwise_accuracy = round(100.0 * np.mean(y_test == (y_hat > 0.5)), 4)

    # Part 3- L2 Regularization
    x_valid = dataset[validation_start:, 0:num_inputs]
    y_valid = dataset[validation_start:, num_inputs]

    # Sweeping lambda
    lbd_opt = 0
    max_log_likelihood = -float("inf")
    resolution = 500 # Sample lambdas in increments of 1/resolution
    lbd_max = 1
    beta_ridge = np.random.randn(num_inputs, 1)
    lambdas = np.linspace(1/resolution, lbd_max, lbd_max*resolution)
    non_changes = 0 # Note that the loss will be convex over lambda so stop testing if we see too many decreases
    print("Training L2...\n")
    
    for l in lambdas:
        beta = SGD_L2(x_train, y_train, beta_init, alpha, l)
        loss = compute_base_loss(x_valid, y_valid, beta) - l*np.sum(np.power(np.abs(beta),2))
        if(loss > max_log_likelihood):
            non_changes = 0
            max_log_likelihood = loss
            lbd_opt = l
            beta_ridge = beta
            print("lambda = " + str(l) + " loss: " + str(loss))
        else:
            non_changes += 1
            if(non_changes > 10):
                break
    print("Optimal lambda: "+str(lbd_opt)+" and loss: "+ str(max_log_likelihood))
    
    y_hat = sigmoid(beta_ridge, x_test.T)
    ridge_accuracy = round(100.0 * np.mean(y_test == (y_hat > 0.5)), 4)

    # Strech Goal 1- Lasso Regularization
    lbd_opt = 0
    max_log_likelihood = -float("inf")
    resolution = 500 # Sample lambdas in increments of 1/resolution
    lbd_max = 1
    beta_lasso = np.random.randn(num_inputs, 1) # Optimal betas
    betas = np.zeros((lbd_max * resolution, num_inputs))
    lambdas = np.linspace(1/resolution, lbd_max, lbd_max*resolution)
    i = 0 # Index into betas
    i_opt = 0 # Python is ridiculous and somehow updates beta_lasso even when loop condition isn't satisfied. This is a stupid workaround for a stupid language!!!! >:[

    print("\nTraining L1...\n")
    for l in lambdas:
        beta = SGD_L1(x_train, y_train, beta_init, alpha, l)
        betas[i,:] = np.atleast_2d(beta).T
        loss = compute_base_loss(x_valid, y_valid, beta) - l*np.sum(np.abs(beta))
        if(loss > max_log_likelihood):
            max_log_likelihood = loss
            lbd_opt = l
            i_opt = i
            beta_lasso = beta
            print("lambda = " + str(l) + " loss: " + str(loss))
        i += 1
    print("Optimal lambda: "+str(lbd_opt)+" and loss: "+ str(max_log_likelihood))
    plt.figure()
    plt.plot(np.linspace(1/resolution, lbd_max, resolution * lbd_max), betas[:, 1:])
    plt.title("Banknote Authenticity Lasso Regression Weights")
    plt.xlabel("lambda")
    plt.ylabel("Predictor Weight")
    plt.axvline(x=lbd_opt, color="black", linestyle="--")
    plt.legend(data_predictors)
    plt.savefig("banknote_lasso_plot.png")

    beta_lasso = betas[i_opt,:]
    y_hat = sigmoid(beta_lasso, x_test.T)
    lasso_accuracy = round(100.0 * np.mean(y_test == (y_hat > 0.5)), 4)
    lasso_selected_predictors = []
    for j in range(1, len(beta_lasso)):
        if beta_lasso[j] != 0:
            lasso_selected_predictors.append(data_predictors[j-1])

    # Results!
    print("\nResults\n")
    print("Stepwise selected predictors: " + str(selected_predictors))
    print("Lasso selected predictors: " + str(lasso_selected_predictors))

    accuracies = {
        "Baseline": baseline_accuracy,
        "No regularization": noreg_accuracy,
        "Stepwise selection": stepwise_accuracy,
        "L2 regularization": ridge_accuracy,
        "L1 regularization": lasso_accuracy,
    }
    acc_table = pd.DataFrame(data=accuracies, index = ["Accuracy (%)"])
    print("Test data Accuracy by Model")
    print(acc_table)

# np.random.seed(32570957363)

main()
