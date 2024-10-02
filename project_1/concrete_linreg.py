# Jeffrey Wong | ECE-472 | Project #1
# Linear Regression - Concrete Strength Data

# The dataset in concrete_data.csv gives the compressive strength of concrete (in MPa)
# as a function of various components and the amount of time it is aged for
# Data got from https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as lmod

np.random.seed(3257095732)

data_predictors = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "plasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]
dataset = np.loadtxt("concrete_data.csv", delimiter=",")
dataset = np.reshape(dataset, newshape=(-1, 9))
dataset_mses = np.zeros((3, 2))
# Make sure to add a Nx1 column of ones to calculate our bias
bias = np.ones((len(dataset), 1))
dataset = np.concatenate((bias, dataset), axis=1)
# Partition dataset
np.random.shuffle(dataset)
testset_start = int(np.floor(0.8 * len(dataset)))
validation_start = int(np.floor(0.9 * len(dataset)))

x_train = dataset[0:testset_start, 0:9]
y_train = dataset[0:testset_start, 9]
x_test = dataset[testset_start:validation_start, 0:9]
y_test = dataset[testset_start:validation_start, 9]

# Part 1 - Standard Linear Regression
beta_hat = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train
y_hat = x_test @ beta_hat
dataset_mses[0, 0] = np.mean(np.power((y_train - x_train @ beta_hat), 2))
dataset_mses[0, 1] = np.mean(np.power((y_test - y_hat), 2))
inv_std_matrix = np.diag(np.power(np.var(dataset[:, 1:9], axis=0), -0.5))
# Based on https://en.wikipedia.org/wiki/Covariance_matrix
cov_matrix = np.cov(
    dataset[:, 1:9].T, bias=True
)  # np.cov treats rows as predictors instead of columns
rho_matrix = (
    inv_std_matrix @ cov_matrix @ inv_std_matrix
)  # The "correlation" matrix is a table of rho_{i,j}s

# Display correlation tables
print("\nCorrelation Coefficients for Prostate Cancer Predictors")
corr_table = pd.DataFrame(rho_matrix, columns=data_predictors, index=data_predictors)
print(corr_table)

# Compute z-scores
label_var = np.sum(
    np.power(dataset[:testset_start, 9] - (dataset[:testset_start, 0:9] @ beta_hat), 2)
) / (testset_start - len(x_test[0]))

beta_sterrs = np.sqrt(
    np.multiply(label_var, np.diag(np.linalg.inv(x_test[:, 1:].T @ x_test[:, 1:])))
)
z_scores = np.divide(beta_hat[1:], beta_sterrs)
z_data = {
    "Coefficient": beta_hat[1:],
    "Std. Err.": beta_sterrs,
    "Z-Score": z_scores,
}
z_table = pd.DataFrame(data=z_data, index=data_predictors)
print("\nZ-scores for Concrete Strength Predictors (excl. bias)")
print(z_table)

# Part 2 - Ridge Regression

# Normalize inputs for parts 2 and 3

y_valid = dataset[validation_start:, 9]
dataset[:, 1:9] = np.divide(
    dataset[:, 1:9] - np.mean(dataset[:, 1:9], axis=0),
    np.sqrt(np.var(dataset[:, 1:9], axis=0)),
)

# We don't need to include the bias term here this time
x_train = dataset[:testset_start, 1:9]
x_test = dataset[testset_start:validation_start, 1:9]
x_valid = dataset[validation_start:, 1:9]

resolution = 200  # How many increments to do per unit
lambda_max = 10
i_max = resolution * lambda_max + 1

betas = np.zeros((i_max, 9))
betas[:, 0] = np.mean(dataset[:testset_start, 9])  # First column represents bias term
# Compute bias based on mean of the labels
print("Sweeping for optimal lambda for ridge regression")
i_opt = 0
mse_min = float("inf")
for i in range(i_max):  # Sweep lambda from 0 to 10 in increments of 1/resolution
    lbd = i / resolution
    # Implements beta = (X^T*X + lambda*I_p)^-1*X^T*Y
    beta_ridge = (
        np.linalg.inv((x_train.T @ x_train) + lbd*np.eye(len(x_train[0])))
        @ x_train.T
        @ y_train
    )
    y_est = x_valid @ beta_ridge + betas[i, 0]
    mse = np.mean(np.power((y_valid - y_est), 2)) + lbd / len(y_valid) * (
        beta_ridge.T @ beta_ridge
    )

    betas[i, 1:] = beta_ridge
    if mse < mse_min:
        mse_min = mse
        i_opt = i
        print("lambda = %f, MSE = %f" % (lbd, mse))
lambda_opt = i_opt / resolution

plt.figure()
plt.plot(np.linspace(0, lambda_max, i_max), betas[:, 1:])
plt.title("Concrete Strength Ridge Regression Weights")
plt.xlabel("lambda")
plt.ylabel("Predictor Weight")
plt.axvline(x=lambda_opt, color="red", linestyle="--")
plt.legend(data_predictors)
plt.savefig("concrete_ridge_plot.png")

dataset_mses[1, 0] = np.mean(np.power(
    (y_train - dataset[:testset_start, 0:9] @ betas[i_opt]), 2)
)
dataset_mses[1, 1] = np.mean(np.power(
    (y_test - dataset[testset_start:validation_start, 0:9] @ betas[i_opt]), 2)
)

# Part 3 - Lasso Regression
resolution = 100
lambda_max = 10
i_max = resolution * lambda_max
betas = np.zeros((i_max, 9))
betas[:, 0] = np.mean(dataset[:testset_start, 9])  # First column represents bias term
print("Sweeping for optimal lambda for lasso regression")
i_opt = 0
mse_min = float("inf")
for i in np.arange(1, i_max + 1):
    lbd = i / resolution
    lasso_reg = lmod.Lasso(alpha=(i / resolution))
    lasso_reg.fit(x_train, y_train)
    beta_lasso = lasso_reg.coef_
    y_est = x_valid @ beta_lasso + betas[i - 1, 0]
    mse = np.mean(np.power((y_valid - y_est), 2)) + lbd / len(y_valid) * np.sum(
        np.abs(beta_lasso)
    )
    betas[i - 1, 1:] = beta_lasso
    if mse < mse_min:
        mse_min = mse
        i_opt = i
        print("lambda = %f, MSE = %f" % (lbd, mse))
lambda_opt = i_opt / resolution

plt.figure()
plt.plot(np.linspace(0.1, lambda_max, i_max), betas[:, 1:])
plt.title("Concrete Strength Lasso Regression Weights")
plt.xlabel("lambda")
plt.ylabel("Predictor Weight")
plt.axvline(x=lambda_opt, color="black", linestyle="--")
plt.legend(data_predictors)
plt.savefig("concrete_lasso_plot.png")

dataset_mses[2, 0] = np.mean(np.power(
    (y_train - dataset[:testset_start, 0:9] @ betas[i_opt]), 2)
)
dataset_mses[2, 1] = np.mean(np.power(
    (y_test - dataset[testset_start:validation_start, 0:9] @ betas[i_opt]), 2)
)

# Display table of training and test MSEs
mse_table = pd.DataFrame(
    data=dataset_mses,
    columns=["Training MSE", "Test MSE"],
    index=["Linear", "Ridge", "Lasso"],
)
print("\nVariance for labels (Reference MSE): " + str(np.var(dataset[:, 9])))
print("\nTraining and Test MSEs for models")
print(mse_table)
