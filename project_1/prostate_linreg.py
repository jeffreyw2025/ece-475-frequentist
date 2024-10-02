# Jeffrey Wong | ECE-472 | Project #1
# Linear Regression - Prostate Cancer Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as lmod

np.random.seed(3257095732)

# Initialize dataset and subset indices
dataset = np.fromfile("prostate.data", sep="\t")
dataset = np.reshape(dataset, newshape=(-1, 10))
data_predictors = [
    "lcavol",
    "lweight",
    "age",
    "lbph",
    "svi",
    "lcp",
    "gleason",
    "pgg45",
]
# Make sure to add a Nx1 column of ones to calculate our bias
dataset[:, 0] = 1  # Initialize first column to all 1s to represent bias term
# The middle 8 columns are predictors, the last column are our labels, lpsa
np.random.shuffle(dataset)
testset_start = int(np.floor(0.8 * len(dataset)))
validation_start = int(np.floor(0.9 * len(dataset)))

x_train = dataset[0:testset_start, 0:9]
y_train = dataset[0:testset_start, 9]
x_test = dataset[testset_start:validation_start, 0:9]
y_test = dataset[testset_start:validation_start, 9]

# Part 1 - Standard Linear Regression
# Implements b = (X^T*X)^-1*X^T*Y
beta_hat = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train
y_hat = x_test @ beta_hat
mse = np.mean(np.power((y_test - y_hat), 2))
print("Prostate Cancer MSE using Linear Regression: ", mse)
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
print("\nZ-scores for Prostate Cancer Predictors (excl. bias)")
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
plt.title("Prostate Cancer Ridge Regression Weights")
plt.xlabel("lambda")
plt.ylabel("Predictor Weight")
plt.axvline(x=lambda_opt, color="red", linestyle="--")
plt.legend(data_predictors)
plt.savefig("prostate_ridge_plot.png")

# Part 3 - Lasso Regression
resolution = 200
lambda_max = 2
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
plt.title("Prostate Cancer Lasso Regression Weights")
plt.xlabel("lambda")
plt.ylabel("Predictor Weight")
plt.axvline(x=lambda_opt, color="black", linestyle="--")
plt.legend(data_predictors)
plt.savefig("prostate_lasso_plot.png")
