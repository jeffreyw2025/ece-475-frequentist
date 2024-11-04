Jeffrey Wong | ECE-475 | Project #4
Gradient Boosted Trees - Project Write-Up

Note: When running thus program, please run them on the modified data file I have attached, hepatitis.data. 
If you want to, you can also comment out the seeding part of the program for different results,
or uncomment the joint dependency plots if you want to verify that they are produced correctly.

Plots I produced can be found in the plots folder, with plots relating to part 1 prefixed with cahousing
and those relating to part 2 (the self-selected dataset) being prefixed with hep.

General Analysis of Hepatitis Data

Immediately, we note that there is a fairly high baseline error rate on this set when using XGBoost:
according to the UCI Repository the baseline accuracy using XGBoost is only 79.5%. We also see a large
amount of variance in the results due to both random chance and based on how we measure feature importance.
For example, sex tends to be a lot higher in importance when considering cover than when considering gain.
The traits that tend to appear most often in the top seven are sex, age, fatigue, ascites, bilirubin, spiders, and varices.

Sex being an important predictor makes a lot of sense- men are significantly more likely to contract both
Hepatitis A and Hepatitis B than women [1]. It also happens that hepatitis is more prevalent among younger
people than in older populations, although the peak is in the 30-39 range. Fatigue also makes sense as it is a symptom of hepatitis. Another
symptom of hepatitis is an abnormally high bilirubin level, which tracks with it being observed as a significant predictor.
Ascites being one of the most prevalent predictors is somewhat odd, as although it can be caused by hepatitis it is usually
caused by cancer or other liver conditions, particularly cirrhosis [3]. Finally, esophageal varices are caused by portal hypertension,
which is often caused by hepatitis.

We can also observe from our joint dependency plot that fatigue and malaise both make it more likely that someone
has hepatitis, and that likelihood is maximized when they have both. What is a little odd is just how straight our lines are, 
but that can be attributed to a low sample size and the categorical nature of our data.

Tl;dr: A lot of the predictors with high association are either sex, which is because men are more likely to contract hepatitis,
or related to hepatitis symptoms. The more symptoms one has, the more one is to have hepatitis.

Shapley Analysis

The Shapley analysis tends to not place a lot of weight on sex, but rather on low protime levels in particular. This is actually
slightly problematic, as protime in particular is not something that is directly associated with hepatitis, but only
tangentially related to general liver health [5]. This anomalous behaviour is likely caused by the significant amount of
samples with no documented protime levels, as the test for this was likely skipped for a lot of these patients, as well as it
having a particularly strong effect only when protime levels were low. The presence of missing values also likely contributed to 
the "swinginess" of certain predictors, which maintained similar magnitudes but could skew either wildly positive or negative 
depending on the sample drawn. Despite these shortcomings, we can still see a fair bit of discrimination along whether patients 
had certain symptoms or not- with those in each class clustering close together and far from the other class.

[1] Centers for Disease Control and Prevention. Viral Hepatitis Surveillance Report – United States, 2022. 
https://www.cdc.gov/hepatitis/statistics/2022surveillance/index.htm Published April 2024. Accessed November 2, 2024.

[2] Medline Plus - National Library of Medicine. Bilirubin blood test. https://medlineplus.gov/lab-tests/bilirubin-blood-test/
Accessed November 2, 2024.

[3] National Cancer Institute. Ascites and Cancer. https://www.cancer.gov/about-cancer/treatment/side-effects/ascites
Published October 2024. Accessed November 2, 2024.

[4] National Library of Medicine. Esophageal varices. https://www.ncbi.nlm.nih.gov/books/NBK448078/
Published January 2024. Accessed November 2, 2024.

[5] Medline Plus - National Library of Medicine. Prothrombin Time Test and INR (PT/INR).
https://medlineplus.gov/lab-tests/prothrombin-time-test-and-inr-ptinr/ Accessed November 2, 2024.