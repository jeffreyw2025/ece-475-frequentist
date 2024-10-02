Jeffrey Wong | ECE-472 | Project #1
Linear Regression - Project Write-Up

Note: When running these programs, please run them on the modified data files I have attached,
prostate.data and concrete_data.csv, respectively. If you want to, you can also comment out the seeding
part of the program for different (and probably even worse) results.

In the process of completing this project, I became familiar with several important numerical and 
machine learning libraries, such as numpy, pandas, and scikit-learn. There is a lot of similarity to,
but not a one-to-one correspondence, with MATLAB syntax, so parsing the documentation alone took some
amount of time. In terms of the results of the program, the plots produced by the prostate data tended
to vary substantially, as we had a small dataset of only 97 values to work off of, meaning that shuffling
the data could result in wildly different parameters and MSEs becoming significant. There also seems to be 
an anomaly where MSEs remain lowest for lambda = 0 for the ridge and lasso regressions in both this dataset
and the concrete dataset, meaning lambda gets stuck at a minimum value most of the time. Verifying with others, 
some encountered other issues with getting a reasonable lambda, so it seems like results can be up to chance.

Meanwhile, for the concrete dataset, the values are inherently much larger, so a higher MSE was somewhat inevitable.
What is strange is that the ridge and lasso models actually generalized worse than baseline linear regression,
as shown by the higher MSEs on the test set, but at least they all performed better than just estimating the mean-
the MSEs are much lower than the label variances. In terms of the most significant predictors, both the Z-scores
and lasso suggest that the amount of cement is by far the most import variable, followed by the presence of slag and
the amount of time the concrete was aged for. Since cement is the principal component of concrete, this makes sense!