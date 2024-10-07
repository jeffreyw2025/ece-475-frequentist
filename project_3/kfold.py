# Jeffrey Wong | ECE-472 | Project #3
# Model Validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN_Classifier

def compute_highest_corrs(X, Y, num_predictors):
    cross_corrs = X.T @ Y
    sorted_predictors = np.sort(np.abs(cross_corrs), axis=0)
    top_predictors = (np.abs(cross_corrs) >= sorted_predictors[-num_predictors]) * 1
    return top_predictors[:].nonzero()

def bad_Kfold(X, Y, N, K):
    top_predictors = compute_highest_corrs(X, Y, 100)
    X = X[:,top_predictors]
    X = X[:,0,:] # Collapse extra dimension
    Y_est = np.zeros(np.shape(Y))
    for k in range(K):
        mask = np.ones((1,N))
        mask[:,N//K*k:N//K*(k+1)] = 0
        mask = (mask == 1)[0,:]
        X_test = X[mask,:]
        Y_test = Y[mask]
        X_valid = X[N//K*k:N//K*(k+1),:]
        Y_valid = Y[N//K*k:N//K*(k+1)]
        knn = KNN_Classifier().fit(X_test, np.ravel(Y_test)) # KNN whines if you pass in a vector instead of an array
        Y_est[N//K*k:N//K*(k+1)] = knn.predict(X_valid)
    accuracy = round(100.0 * np.sum(Y_est == Y) / len(Y), 2)
    return accuracy

def good_Kfold(X, Y, N, K):
    Y_est = np.zeros(np.shape(Y))
    for k in range(K):
        mask = np.ones((1,N))
        mask[:,N//K*k:N//K*(k+1)] = 0
        mask = (mask == 1)[0,:]
        X_test = X[mask,:]
        Y_test = Y[mask]
        X_valid = X[N//K*k:N//K*(k+1),:]
        Y_valid = Y[N//K*k:N//K*(k+1)]
        # In the correct way you only compute correlations based on the test data
        top_predictors = compute_highest_corrs(X_test, Y_test, 100)
        X_test = X_test[:,top_predictors]
        X_test = X_test[:,0,:] # Collapse extra dimension
        X_valid = X_valid[:,top_predictors]
        X_valid = X_valid[:,0,:] # Collapse extra dimension
        knn = KNN_Classifier().fit(X_test, np.ravel(Y_test))
        Y_est[N//K*k:N//K*(k+1)] = knn.predict(X_valid)
    accuracy = round(100.0 * np.sum(Y_est == Y) / len(Y), 2)
    return accuracy

def main():
    # Generate initial data
    N = 50
    K = 5 # We will just do 5-fold cross validation
    p = 5000
    X_test = np.random.randn(N,p) # Predictors are independent standard Gaussian
    Y_test = np.r_[np.zeros((N//2,)), np.ones((N//2,))]
    np.random.shuffle(Y_test)
    print("Doing \"Incorrect\" K-fold")
    bad_Kfold_accuracy = bad_Kfold(X_test, Y_test, N, K)
    print("Doing \"Correct\" K-fold")
    good_Kfold_accuracy = good_Kfold(X_test, Y_test, N, K)
    accuracies = {
        "Expected": 50.0, # Everything is 50-50: either it happens or it doesn't
        "\"Incorrect\" Method": bad_Kfold_accuracy, # Should be far from expected accuracy
        "\"Correct\" Method": good_Kfold_accuracy, # Should be close to the expected value
    }
    acc_table = pd.DataFrame(data=accuracies, index = ["Accuracy (%)"])
    print("Test data Accuracy by Model")
    print(acc_table)

main()