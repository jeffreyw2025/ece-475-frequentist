# Jeffrey Wong | ECE-475 | Project #4
# Gradient Boosted Trees
# Hepatitis Dataset found at https://archive.ics.uci.edu/dataset/46/hepatitis

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import shap

def california_housing():
    # For the California Housing dataset we will be analyzing:
    # 1) Average absolute error over iterations
    # 2) (relative) Feature importance
    # 3) Partial dependence on median income, average occupancy, house age, and average rooms
    # 4) Joint partial dependence on house age and average occupancy

    data_predictors = [
        "income", # 1
        "house_age", # 2
        "avg_rooms", # 3
        "avg_bedrooms", # 4
        "avg_occupancy", # 5
        "households", # 6
        "latitude", # 7
        "longitude", # 8
    ]

    dataset = np.fromfile("cadata.txt", sep = "  ")
    dataset = np.reshape(dataset, newshape=(-1, 9))
    dataset[:, 0] = np.divide(dataset[:, 0], 1e5) # Textbook measures in units of 100k
    dataset[:, 3] = np.divide(dataset[:, 3], dataset[:, 6]) # Divide total rooms/bedrooms by households to get average rooms/bedrooms per household
    dataset[:, 4] = np.divide(dataset[:, 4], dataset[:, 6])
    dataset[:, 5] = np.divide(dataset[:, 5], dataset[:, 6]) # Divide total pop by households to get average household occupancy

    np.random.shuffle(dataset)
    testset_start = int(np.floor(0.8 * len(dataset)))

    # Our predictor, the median price, is the first column
    x_train = dataset[0:testset_start, 1:]
    y_train = dataset[0:testset_start, 0]
    x_test = dataset[testset_start:, 1:]
    y_test = dataset[testset_start:, 0]

    dtrain = xgb.DMatrix(x_train, label = y_train, feature_names = data_predictors)
    dtest = xgb.DMatrix(x_test, label = y_test, feature_names = data_predictors)

    # From the textbook:
    # Parameters: We fit a gradient boosting model using the MART procedure, with J = 6
    # terminal nodes, a learning rate of ν = 0.1, and the Huber loss
    # criterion for predicting the numeric response
    params = {
        "eta": 0.1,
        "max_leaves": 6,
        "objective": "reg:pseudohubererror",
        "eval_metric": "mae",
    }
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    losses = {}
    num_rounds = 1001
    reg = xgb.XGBRegressor(
        n_estimators = num_rounds,
        max_leaves=params["max_leaves"], 
        learning_rate=params["eta"], 
        objective=params["objective"],
        eval_metric=params["eval_metric"],
    )
    reg.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose = False)

    bst = xgb.train(params, dtrain, num_rounds, evals=evallist, evals_result = losses, verbose_eval = 50)

    plt.figure()
    plt.plot(np.arange(num_rounds), losses["train"]["mae"])
    plt.plot(np.arange(num_rounds), losses["eval"]["mae"])
    plt.title("Mean Absolute Error on Training and Test Data")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Absolute Error")
    plt.legend(["Train", "Test"])
    plt.savefig("cfhousing_error.png")

    plt.figure()
    ax_importance = xgb.plot_importance(bst, grid = False, label = data_predictors, xlabel = "Average Gain", importance_type='gain')
    plt.savefig("cfhousing_importance.png")

    plt.figure()
    # Partial dependency display handles the decile marks on the x-axis automatically
    PartialDependenceDisplay.from_estimator(
        reg, 
        x_test, 
        ["income", "house_age", "avg_rooms", "avg_occupancy"], 
        n_cols=2, 
        feature_names= data_predictors
    )
    plt.suptitle("Individual Partial Dependency Plots")
    plt.tight_layout()

    plt.savefig("cfhousing_indep_pardep.png")

    # Joint partial dependency plots are sloooow. Uncomment at your discretion.

    plt.figure()
    PartialDependenceDisplay.from_estimator(
        reg, 
        x_test, 
        [("house_age", "avg_occupancy"),], 
        n_cols=1, 
        feature_names= data_predictors
    )
    plt.title("Joint Partial Dependency Plot of House Age and Occupancy")
    plt.savefig("cfhousing_joint_pardep.png")

    plt.figure()
    PartialDependenceDisplay.from_estimator(
        reg, 
        x_test, 
        [("latitude", "longitude"),], 
        n_cols=1, 
        feature_names= data_predictors
    )
    plt.title("Joint Partial Dependency Plot of Latitude and Longitude")
    plt.savefig("cfhousing_pos_pardep.png")

def alternate_dataset():
    data_predictors = [
        "age", # 1, integer
        "sex", # 2, binary- unfortunately
        "steroid", # 3, binary
        "antivirals", # 4, binary
        "fatigue", # 5, binary
        "malaise", # 6, binary
        "anorexia", # 7, binary
        "liver_big", # 8, binary
        "liver_firm", # 9, binary
        "spleen_palpable", # 10, binary
        "spiders", # 11, binary
        "ascites", # 12, binary
        "varices", # 13, binary
        "bilirubin", # 14, continuous
        "alk_phosphate", # 15, integer
        "sgot", # 16, integer
        "albumin", # 17, integer
        "protime", # 18, integer
        "histology", # 19, integer
    ]
    predictor_types = ["int", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "q", "i", "i", "i", "i", "i"]
    dataset = np.genfromtxt("hepatitis.data", delimiter = ",") # genfromtxt handles missing values and defaults them to NaN
    dataset = np.reshape(dataset, newshape=(-1, 20))
    dataset[:, 0] = dataset[:, 0] - 1 # Convert classes to 0/1

    np.random.shuffle(dataset)
    testset_start = int(np.floor(0.8 * len(dataset)))

    x_train = pd.DataFrame(data = dataset[0:testset_start, 1:], columns = data_predictors)
    y_train = pd.DataFrame(data = dataset[0:testset_start, 0])
    x_test = pd.DataFrame(data = dataset[testset_start:, 1:], columns = data_predictors)
    y_test = pd.DataFrame(data = dataset[testset_start:, 0])

    # XGBoost has special handling for categorical data, need to specify predictor types to leverage this capability
    dtrain = xgb.DMatrix(x_train, label = y_train, feature_names = data_predictors, feature_types = predictor_types)
    dtest = xgb.DMatrix(x_test, label = y_test, feature_names = data_predictors, feature_types = predictor_types)

    print("Fraction of parameters missing in training data: " + str((np.prod(np.shape(x_train)) - dtrain.num_nonmissing())/np.prod(np.shape(x_train))))
    print("Fraction of parameters missing in test data: " + str((np.prod(np.shape(x_test)) - dtest.num_nonmissing())/np.prod(np.shape(x_test))))

    params = {
        "eta": 0.03,
        "max_leaves": 6,
        "objective": "binary:logistic",
        "eval_metric": "error",
        "min_child_weight": 1,
    }
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    losses = {}
    num_rounds = 301
    
    clf = xgb.XGBClassifier(
        n_estimators = num_rounds,
        max_leaves=params["max_leaves"], 
        learning_rate=params["eta"], 
        objective=params["objective"],
        eval_metric=params["eval_metric"],
    )
    clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose = False)

    bst = xgb.train(params, dtrain, num_rounds, evals=evallist, evals_result = losses, verbose_eval = 20)

    plt.figure()
    plt.plot(np.arange(num_rounds), losses["train"]["error"])
    plt.plot(np.arange(num_rounds), losses["eval"]["error"])
    plt.title("Mean Absolute Error on Training and Test Data")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Absolute Error")
    plt.legend(["Train", "Test"])
    plt.savefig("hep_error.png")

    plt.figure()
    ax_importance = xgb.plot_importance(bst, grid = False, xlabel = "Average Cover", importance_type='cover')
    plt.savefig("hep_importance.png")

    plt.figure()
    # Partial dependency display handles the decile marks on the x-axis automatically
    PartialDependenceDisplay.from_estimator(
        clf, 
        x_test, 
        ["sex", "age", "fatigue", "ascites", "varices", "bilirubin"], 
        n_cols=3, 
        feature_names= data_predictors
    )
    plt.suptitle("Individual Partial Dependency Plots")
    plt.tight_layout()
    plt.savefig("hep_indep_pardep.png")

    plt.figure()
    plt.suptitle("Joint Partial Dependency Plot of Fatigue and Malaise")
    PartialDependenceDisplay.from_estimator(
        clf, 
        x_test, 
        [("fatigue", "malaise"),], 
        n_cols=1, 
        feature_names= data_predictors
    )
    plt.savefig("hep_joint_pardep.png")

    # Shapely stuff for extra credit

    # Generate some basic plots- waterfall shows relative importance and direction
    explainer = shap.Explainer(clf, x_train)
    shap_values_xgb = explainer(x_test)
    plt.figure()
    shap.plots.waterfall(shap_values_xgb[20][:])
    plt.savefig("hep_shap_waterfall.png")
    
    # Beeswarm shows impact of individual samples
    plt.figure()
    shap.plots.beeswarm(shap_values_xgb)
    plt.savefig("hep_shap_beeswarm.png")

    plt.figure()
    shap.plots.scatter(shap_values_xgb[:, "protime"], color=shap_values_xgb[:, "protime"])
    plt.savefig("hep_shap_scatter.png")

def main():
    print("It never rains in Southern California")
    # california_housing()
    print("Analyzing Hepatitis dataset")
    alternate_dataset() # Hepatitis dataset

if __name__ == "__main__":
    main()
