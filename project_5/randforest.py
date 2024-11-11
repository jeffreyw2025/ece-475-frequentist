# Jeffrey Wong | ECE-475 | Project #5
# Random Forests

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as ttsplit
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def calculate_aae(rfr, x_test, y_test):
    y_hat = rfr.predict(x_test)
    aae = np.mean(np.abs(y_hat - y_test))
    return aae

def calculate_rmse(rfr, x_test, y_test):
    y_hat = rfr.predict(x_test)
    rmse = np.sqrt(np.mean(np.power(np.abs(y_hat - y_test),2)))
    return rmse

def california_housing():
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

    # Preprocessing data
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
    # Two random forests are shown, with
    # m = 2 and m = 6. The two gradient boosted models use a shrinkage parameter
    # ν = 0.05 in (10.41), and have interaction depths of 4 and 6.
    num_rounds = 1001
    forest_resolution = 10 # Number of trees to add to RF at a time
    xgb4_params = {
        "eta": 0.05,
        "max_depth": 4,
        "objective": "reg:pseudohubererror",
        "eval_metric": "mae",
    }
    xgb6_params = {
        "eta": 0.05,
        "max_depth": 6,
        "objective": "reg:pseudohubererror",
        "eval_metric": "mae",
    }
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    losses_rfr2 = np.zeros((num_rounds // forest_resolution,))
    losses_rfr6 = np.zeros((num_rounds // forest_resolution,))
    losses_xgb4 = {}
    losses_xgb6 = {}

    print("Training Random Forests")
    rfr_6 = RandomForestRegressor(
        n_estimators = 250, 
        criterion="absolute_error",
        max_depth = 10,
        max_features=6,
        max_samples=0.75,
        n_jobs = -1 # Train using all the processors
    ).fit(x_train, y_train)
    """
    for n_tree in np.arange(forest_resolution, num_rounds, forest_resolution):
        n_iter = n_tree//forest_resolution
        print("Iteration "+ str(n_iter)+ "; Num trees: " + str(n_tree))
        
        rfr_2 = RandomForestRegressor(
            n_estimators = n_tree, 
            criterion="absolute_error",
            max_depth = 10,
            max_features=2,
            max_samples=0.75,
            n_jobs = -1 # Train using all the processors
        ).fit(x_train, y_train)
        rfr_6 = RandomForestRegressor(
            n_estimators = n_tree, 
            criterion="absolute_error",
            max_depth = 10,
            max_features=6,
            max_samples=0.75,
            n_jobs = -1 # Train using all the processors
        ).fit(x_train, y_train)
        mae_2 = calculate_aae(rfr_2, x_test, y_test)
        mae_6 = calculate_aae(rfr_6, x_test, y_test)
        losses_rfr2[n_iter-1] = mae_2
        losses_rfr6[n_iter-1] = mae_6
        print("Loss for m = 2: "+ str(mae_2) +"; m = 6: "+ str(mae_6))
    """
    print("Training GB Trees to Depth 4")
    bst_xgb4 = xgb.train(xgb4_params, dtrain, num_rounds, evals=evallist, evals_result = losses_xgb4, verbose_eval = 50)
    print("Training GB Trees to Depth 6")
    bst_xgb6 = xgb.train(xgb6_params, dtrain, num_rounds, evals=evallist, evals_result = losses_xgb6, verbose_eval = 50)

    plt.figure()
    plt.plot(np.arange(forest_resolution, num_rounds, forest_resolution), losses_rfr2)
    plt.plot(np.arange(forest_resolution, num_rounds, forest_resolution), losses_rfr6)
    plt.plot(np.arange(num_rounds), losses_xgb4["eval"]["mae"])
    plt.plot(np.arange(num_rounds), losses_xgb6["eval"]["mae"])
    plt.title("Mean Absolute Error on California Housing Test Data")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Absolute Error")
    # plt.ylim((0.25, 0.5))
    plt.legend(["RF m = 2", "RF m = 6", "GBT depth = 4", "GBT depth = 6"])
    plt.savefig("cfhousing_error.png")

    plt.figure()
    xgb.plot_importance(bst_xgb6, grid = False, title = "Importance with Gradient Boosted Trees", label = data_predictors, xlabel = "Average Cover", importance_type='gain')
    plt.savefig("cfhousing_importance_gbtrees.png")

    # Permutation importance plot code taken from https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    result = permutation_importance(rfr_6, x_test, y_test, n_jobs=-1)
    forest_importances = pd.Series(result.importances_mean, index=data_predictors)

    fig, ax = plt.subplots()
    # xgb.plot_importance(bst_rfr6_alt, grid = False, title = "Importance with Random Forests", label = data_predictors, xlabel = "Average Cover", importance_type='gain')
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Importance with Random Forests")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig("cfhousing_importance_rndfst.png")

def seoul_bikes():
    START_DATE = pd.Timestamp(year=2017, month=12, day=1)
    data_predictors = [
        "days_since_start",
        "hour",
        "temp",
        "humidity",
        "wind_speed",
        "10m_vis",
        "dew_point",
        "solar_radiation",
        "rainfall",
        "snowfall",
        "is_winter",
        "is_spring",
        "is_summer",
        "is_autumn",
        "is_holiday",
    ]
    dataset = pd.read_csv("SeoulBikeData.csv", delimiter=",", names = ["num_bikes", "date"] + data_predictors[1:] + ["functional_day"], parse_dates = [1], date_format = "%d/%m/%Y")
    # Remove non-functional times where no bikes were used
    dataset = dataset.loc[dataset["functional_day"] == 1]

    # Convert dates to days since start date (12/1/17)
    # Code taken from https://stackoverflow.com/questions/26072087/pandas-number-of-days-elapsed-since-a-certain-date
    
    dataset['days_since_start'] = (dataset['date'] - START_DATE).dt.days

    # Partition into test and train sets
    x = dataset[data_predictors]
    y = dataset["num_bikes"]
    x_train, x_test, y_train, y_test = ttsplit(x, y, test_size=0.2)
    dtrain = xgb.DMatrix(x_train, label = y_train, feature_names = data_predictors)
    dtest = xgb.DMatrix(x_test, label = y_test, feature_names = data_predictors)

    num_rounds = 501
    forest_resolution = 10 # Number of trees to add to RF at a time
    xgb5_params = {
        "eta": 0.05,
        "max_depth": 5,
        "gamma": 1,
        "objective": "reg:squarederror",
    }
    xgb8_params = {
        "eta": 0.05,
        "max_depth": 8,
        "gamma": 1,
        "objective": "reg:squarederror",
    }
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    losses_rfr3 = np.zeros((num_rounds // forest_resolution,))
    losses_rfr10 = np.zeros((num_rounds // forest_resolution,))
    losses_xgb5 = {}
    losses_xgb8 = {}

    print("Training Random Forests")
    rfr_10 = 0
    
    for n_tree in np.arange(forest_resolution, num_rounds, forest_resolution):
        n_iter = n_tree//forest_resolution
        print("Iteration "+ str(n_iter)+ "; Num trees: " + str(n_tree))
        
        rfr_3 = RandomForestRegressor(
            n_estimators = n_tree, 
            criterion="squared_error",
            max_depth = 10,
            max_features=3,
            max_samples=0.75,
            n_jobs = -1 # Train using all the processors
        ).fit(x_train, y_train)
        rfr_10 = RandomForestRegressor(
            n_estimators = n_tree, 
            criterion="squared_error",
            max_depth = 10,
            max_features=10,
            max_samples=0.75,
            n_jobs = -1 # Train using all the processors
        ).fit(x_train, y_train)
        rmse_3 = calculate_rmse(rfr_3, x_test, y_test)
        rmse_10 = calculate_rmse(rfr_10, x_test, y_test)
        losses_rfr3[n_iter-1] = rmse_3
        losses_rfr10[n_iter-1] = rmse_10
        print("Loss for m = 3: "+ str(rmse_3) +"; m = 10: "+ str(rmse_10))
    
    print("Training GB Trees to Depth 5")
    bst_xgb5 = xgb.train(xgb5_params, dtrain, num_rounds, evals=evallist, evals_result = losses_xgb5, verbose_eval = 50)
    print("Training GB Trees to Depth 8")
    bst_xgb8 = xgb.train(xgb8_params, dtrain, num_rounds, evals=evallist, evals_result = losses_xgb8, verbose_eval = 50)

    plt.figure()
    plt.plot(np.arange(forest_resolution, num_rounds, forest_resolution), losses_rfr3)
    plt.plot(np.arange(forest_resolution, num_rounds, forest_resolution), losses_rfr10)
    plt.plot(np.arange(num_rounds), losses_xgb5["eval"]["rmse"])
    plt.plot(np.arange(num_rounds), losses_xgb8["eval"]["rmse"])
    plt.title("RMS Error on Seoul Bike Test Data")
    plt.xlabel("Iterations")
    plt.ylabel("RMS Error")
    # plt.ylim((0.25, 0.5))
    plt.legend(["RF m = 3", "RF m = 10", "GBT depth = 5", "GBT depth = 8"])
    plt.savefig("seoulbike_error.png")

    plt.figure()
    xgb.plot_importance(
        bst_xgb8, 
        grid = False, 
        title = "Importance with Gradient Boosted Trees", 
        xlabel = "Average Cover", 
        importance_type='gain',
    )
    plt.savefig("seoulbike_importance_gbtrees.png")

    # Permutation importance plot code taken from https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    result = permutation_importance(rfr_10, x_test, y_test, n_jobs=-1)
    forest_importances = pd.Series(result.importances_mean, index=data_predictors)

    fig, ax = plt.subplots()
    # xgb.plot_importance(bst_rfr6_alt, grid = False, title = "Importance with Random Forests", label = data_predictors, xlabel = "Average Cover", importance_type='gain')
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Importance with Random Forests")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig("seoulbike_importance_rndfst.png")
    return 0

def main():
    print("It pours, man it pours")
    # california_housing()
    print("Analyzing Seoul bike dataset")
    seoul_bikes()

if __name__ == "__main__":
    main()
