# Jeffrey Wong | ECE-472 | HW #1
# Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0x46B21B55)  # Seed for consistent results
# Miscellaneous Function Definitions

# Prostate Cancer

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

# Part 1 - Standard Linear Regression
x_train = dataset[0:testset_start - 1, 0:9]
y_train = dataset[0:testset_start - 1, 9]
# Implements b = (X^T*X)^-1*X^T*Y
beta_hat = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train
x_test = dataset[testset_start:validation_start - 1, 0:9]
y_test = dataset[testset_start:validation_start - 1, 9]
y_hat = x_test @ beta_hat
mse = np.mean(np.power((y_test - y_hat), 2))
print("Prostate Cancer MSE using Linear Regression: ", mse)
inv_std_matrix = np.diag(np.power(np.var(dataset[:, 1:9], axis=0), -0.5))
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

label_var = 1

beta_stdevs = np.divide(np.diag(inv_std_matrix), np.sqrt(label_var))

z_scores = np.multiply(beta_hat[1:], beta_stdevs)

z_data = {
    "Coefficient": beta_hat[1:],
    "Std. Err.": np.power(beta_stdevs, -1),
    "Z-Score": z_scores,
}

z_table = pd.DataFrame(data=z_data, index=data_predictors)
print("\nZ-scores for Prostate Cancer Predictors (excl. bias)")
print(z_table)

# Part 2 - Ridge Regression

# Implements beta = (X^T*X - lambda*I_{p+1})^-1*X^T*Y

# Part 3 - Lasso Regression
