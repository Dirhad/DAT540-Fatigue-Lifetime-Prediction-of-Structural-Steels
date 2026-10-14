# How many bending cycles (fatigue) before the metal fails
# Nf is the number we count up to (Number of cycles until failure)
# N for number (of cycles), f for failure, σa is how hard you bend it (MPa)
# Sigma is stress, a is amplitude

# Important terms:
# RMSE - typical miss in log units
# MAE - average miss without squaring
# R^2 - share of the variation the model explains (1 = perfect)


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Columns that are never used as inputs
DROP_FROM_X = ["log10_Nf", "group", "fold", "runout"]



def load_clean(path="Data/train.csv"):
    # Load Data 1's cleaned split
    df = pd.read_csv(path)
    df = df.rename(columns={"σa": "sigma_a"}) #rename to sigma_a
    X = df.drop(columns=[c for c in DROP_FROM_X if c in df.columns]) # Removes the columns from the table
    y = df["log10_Nf"] # y is what its supposed to guess
    groups = df["group"] 
    folds = df["fold"] if "fold" in df.columns else None 
    return X, y, groups, folds


def compute_metrics(y_true, y_pred):
    # 4 metrics on how wrong a prediction was 
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # Typical miss
    mae = mean_absolute_error(y_true, y_pred) # Mean of average error
    r2 = r2_score(y_true, y_pred) # share of the variation score (1.0 is perfect, 0.0 is worst)

    s = np.abs(y_pred - y_true) # Abs misses per row
    hits = s < np.log10(2) # True where they are within a factor of 2
    within_factor_2 = np.mean(hits) # True counts as 1, so mean = share
    
    return {"rmse": rmse, "mae": mae, "r2": r2, "within_factor_2": within_factor_2}


def self_test():
    # Check so the metrics verify themselves on every run
    y_true = np.array([4.0, 5.0, 6.0])
    perfect = compute_metrics(y_true, y_true)
    assert perfect["rmse"] == 0.0 and perfect["r2"] == 1.0 # Assert crashes the program if the guess is False
    assert perfect["within_factor_2"] == 1.0
    missed = compute_metrics(y_true, np.array([4.0, 5.0, 6.31]))
    assert abs(missed["within_factor_2"] - 2 / 3) < 1e-9


def basquin_baseline(X_train, y_train, X_test):
    # Straight line between log spread and log lifetime - Basquins Law
    log_sigma_train = np.log10(X_train[["sigma_a"]]) 
    log_sigma_test = np.log10(X_test[["sigma_a"]])
    model = LinearRegression().fit(log_sigma_train, y_train)

    return model.predict(log_sigma_test)

def sklearn_fit_predict(model):
    # First fit then predict the different models
    def inner(X_train, y_train, X_test):
        return model.fit(X_train, y_train).predict(X_test)
    return inner

def cross_validate_model(fit_predict, X, y, groups, n_splits = 5):
    gkf = GroupKFold(n_splits=n_splits)
    results = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        y_pred = fit_predict(X_tr, y_tr, X_te)
        m = compute_metrics(y_te, y_pred)
        m["fold"] = fold
        m["n_test"] = len(test_idx)
        results.append(m)
    return pd.DataFrame(results)

def cross_validate_folds(fit_predict, X, y, folds):
    # Five rounds. For every round one bit is test and the 4 others are training bits.
    results = []
    for f in sorted(folds.unique()):
        # folds == f gives a list with True or False, one per row
        # X[te] picks out the rows with True inside
        tr, te = folds != f, folds == f  
        y_pred = fit_predict(X[tr], y[tr], X[te])

        # Train on four bits, and guess the fifth
        m = compute_metrics(y[te], y_pred)

        # Compare the guess with the correct answer which we have hidden
        m["fold"] = f
        m["n_test"] = int(te.sum())
        results.append(m)
    return pd.DataFrame(results)




MODELS = {
    "basquin_baseline": basquin_baseline,
    "ridge": sklearn_fit_predict(make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    "random_forest": sklearn_fit_predict(RandomForestRegressor(n_estimators=300, random_state=0)),
    "hist_gradient_boosting": sklearn_fit_predict(HistGradientBoostingRegressor(random_state=0)),
}


def run_all(X, y, groups, folds=None):
    all_results = []
    for name, fit_predict in MODELS.items():
        if folds is not None:
            res = cross_validate_folds(fit_predict, X, y, folds)
        else:
            res = cross_validate_model(fit_predict, X, y, groups)
        res["model"] = name
        all_results.append(res)
 
    all_results = pd.concat(all_results)
    metrics = ["rmse", "mae", "r2", "within_factor_2"]

    # Group the 20 rows (4 models x 5 folds) after model and calculate 
    return all_results.groupby("model")[metrics].agg(["mean", "std"])



if __name__ == "__main__":
    self_test()

    X, y, groups, folds = load_clean()
 
    print("rows:", len(X), " features:", len(X.columns), " groups:", groups.nunique())
    print()
    print(run_all(X, y, groups, folds).round(3))