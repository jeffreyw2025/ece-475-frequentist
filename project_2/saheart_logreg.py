# Jeffrey Wong | ECE-472 | Project #2
# Logistic Regression- South African Heart Disease Data

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
    losses = np.zeros((num_observations,1))
    for i in range(num_observations):
        for j in range(len(X_train[0])):
            w[j] += learn_rate * (Y_train[i] - sigmoid(w, X_train[i,:])) * X_train[i,j]
        losses[i] = compute_base_loss(X_train, Y_train, w)
    plt.figure()
    plt.plot(np.arange(num_observations), losses)
    plt.title("Unregularized SGD log Likelihood")
    plt.savefig("loss_unreg.png")
    return w

# Part 3- L2 Regularization
def SGD_L2(X_train, Y_train, W_init, learn_rate, lbd):
    w = W_init
    num_observations = len(X_train)
    losses = np.zeros((num_observations,1))
    for i in range(num_observations):
        for j in range(len(X_train[0])):
            ridge_penalty = (0 if j == 0 else lbd * w[j]) # Don't penalize bias!
            w[j] += learn_rate * (Y_train[i] - sigmoid(w, X_train[i,:]) - ridge_penalty) * X_train[i,j]
        losses[i] = compute_base_loss(X_train, Y_train, w)
        losses[i] -= lbd/2 * np.sum(np.power(np.abs(w[1:]),2))
    return (w, losses)

# Strech Goal 1- L1 Regularization
def SGD_L1(X_train, Y_train, W_init, learn_rate, lbd):
    w = W_init
    u = 0 # Cumulative penalty term
    q = np.zeros((1,len(w)))
    num_observations = len(X_train)
    losses = np.zeros((num_observations,1))
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
        losses[i] = compute_base_loss(X_train, Y_train, w)
        losses[i] -= lbd * np.sum(np.abs(w[1:]))
    return (w, losses)

def compute_base_loss(X, Y, w):
    loss = 0
    for k in range(len(X)):
        loss += Y[k]*np.log(sigmoid(w, X[k,:])) + (1-Y[k])*np.log(1-sigmoid(w, X[k,:]))
    return loss

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
                plt.scatter(X[positive_cases, i], X[positive_cases, j], s=0.007, c='#0000ff')
                plt.scatter(X[negative_cases, i], X[negative_cases, j], s=0.007, c='#ff0000')
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

    # Normalize non-intercept predictors so that we're not passing big numbers into the sigmoid
    dataset[:, 1:num_inputs] = np.divide(
        dataset[:, 1:num_inputs] - np.mean(dataset[:, 1:num_inputs], axis=0),
        np.sqrt(np.var(dataset[:, 1:num_inputs], axis=0)),
    )

    # Replicate 4.22- 9*9 grid of plots comparing risk factors and grouping by pos/neg
    # This function is a little slow so commented out for speed- I promise it works!

    # scatter_matrix(data_predictors, dataset[:, 1:num_inputs], dataset[:, num_inputs])

    # Part 1- Basic Logistic Regression
    x_train = dataset[0:testset_start, 0:num_inputs]
    y_train = dataset[0:testset_start, num_inputs]
    x_test = dataset[testset_start:validation_start, 0:num_inputs]
    y_test = dataset[testset_start:validation_start, num_inputs]

    alpha = 0.05
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
        lr, tol=0.01, direction="forward", scoring='accuracy'
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
    losses_opt = 0
    max_log_likelihood = -float("inf")
    resolution = 500 # Sample lambdas in increments of 1/resolution
    lbd_max = 1
    beta_ridge = np.random.randn(num_inputs, 1)
    lambdas = np.linspace(1/resolution, lbd_max, lbd_max*resolution)
    non_changes = 0 # Note that the loss will be convex over lambda so stop testing if we see too many decreases
    print("Training L2...\n")
    
    for l in lambdas:
        (beta, losses) = SGD_L2(x_train, y_train, beta_init, alpha, l)
        loss = compute_base_loss(x_valid, y_valid, beta) - l*np.sum(np.power(np.abs(beta),2))
        if(loss > max_log_likelihood):
            non_changes = 0
            max_log_likelihood = loss
            lbd_opt = l
            beta_ridge = beta
            losses_opt = losses # Want to plot losses of optimal 
            print("lambda = " + str(l) + " loss: " + str(loss))
        else:
            non_changes += 1
            if(non_changes > 10):
                break
    print("Optimal lambda: "+str(lbd_opt)+" and loss: "+ str(max_log_likelihood))
    plt.figure()
    plt.plot(np.arange(len(x_train)), losses_opt)
    plt.title("L2 SGD log Likelihood")
    plt.savefig("loss_ridge.png")
    
    y_hat = sigmoid(beta_ridge, x_test.T)
    ridge_accuracy = round(100.0 * np.mean(y_test == (y_hat > 0.5)), 4)

    # Strech Goal 1- Lasso Regularization
    lbd_opt = 0
    losses_opt = 0
    max_log_likelihood = -float("inf")
    resolution = 200 # Sample lambdas in increments of 1/resolution
    lbd_max = 1
    beta_lasso = np.random.randn(num_inputs, 1) # Optimal betas
    betas = np.zeros((lbd_max * resolution, num_inputs))
    lambdas = np.linspace(1/resolution, lbd_max, lbd_max*resolution)
    i = 0 # Index into betas
    i_opt = 0

    print("Training L1...\n")
    for l in lambdas:
        (beta, losses) = SGD_L1(x_train, y_train, beta_init, alpha, l)
        betas[i,:] = beta.T
        loss = compute_base_loss(x_valid, y_valid, beta) - l*np.sum(np.abs(beta))
        if(loss > max_log_likelihood):
            max_log_likelihood = loss
            lbd_opt = l
            beta_lasso = beta
            losses_opt = losses # Want to plot losses of optimal 
            print("lambda = " + str(l) + " loss: " + str(loss))
            i_opt = i
        i += 1
    print("Optimal lambda: "+str(lbd_opt)+" and loss: "+ str(max_log_likelihood))
    plt.figure()
    plt.plot(np.arange(len(x_train)), losses_opt)
    plt.title("L1 SGD log Likelihood")
    plt.savefig("loss_lasso.png")

    plt.figure()
    plt.plot(np.linspace(1/resolution, lbd_max, resolution * lbd_max), betas[:, 1:])
    plt.title("Heart Disease Lasso Regression Weights")
    plt.xlabel("lambda")
    plt.ylabel("Predictor Weight")
    plt.axvline(x=lbd_opt, color="black", linestyle="--")
    plt.legend(data_predictors)
    plt.savefig("SAheart_lasso_plot.png")

    beta_lasso = betas[i_opt,:]
    y_hat = sigmoid(beta_lasso, x_test.T)
    lasso_accuracy = round(100.0 * np.mean(y_test == (y_hat > 0.5)), 4)
    lasso_selected_predictors = []
    print(beta_lasso)
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
