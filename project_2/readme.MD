Jeffrey Wong | ECE-472 | Project #2
Logistic Regression - Project Write-Up

Note: When running these programs, please run them on the modified data files I have attached,
SAheart.csv, data_banknote_authentication.txt, and iris.data, respectively. If you want to, 
you can also comment out the seeding part of the program for different results.

For this project, I found that I was able to successfully establish a sensible hierarchy among most of the methods-
All regression methods produced an accuracy above the baseline across almost all random tests, with L2 performing around the same
as unregularized and stepwise, being implemented by zeroing out non-selected parameter weights, performing worse
than unregularized, which makes sense. It was found that for the same learning rate, the model was able to converge to a much higher
accuracy on the banknote data than the heart disease data, which makes sense as the former had both more observations and fewer predictors
than the latter, allowing for a much tighter fit. In fact, I had to actually lower the learning rate for the banknote version, and it
still performed better!

The lasso method produces a high accuracy for the SAheart dataset, but selects a slightly low value of lambda and thus includes
a few more parameters than the stepwise selection. It selects a lambda that is way too low for the banknote dataset, though, causing
it to just include everything.

It was rather easy to generalize to the multinomial classification case- the likelihood of each class can be considered independently
then we just use the ML criterion to select the most likely class- there are an equal number of observations with each label. This
is similar to the analysis I conducted in an earlier project for Stochastic Processes. Generally the model has an accuracy around 25%
higher than baseline. One note is that strictly speaking, it would be more accurate to use a MAP criterion on the training dataset, but
since the number of observations of each class should be roughly the same I figured it wouldn't matter too much. There is a bit of an anomaly,
though, as although generally likelihood increases over iterations of SGD it does not always do so consistently. This is likely
because the dataset is so small that permutations that cause pathological behavior are just more likely.