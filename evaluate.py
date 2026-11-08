import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option("display.width", 200) 
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

DROP_FROM_X = ["log10_Nf", "group", "fold", "runout"]


COLLINEAR_BLOCKS = {
    "strength (proof/tensile/vickers)":
        ["0.2% proof stress", "Tensile strength", "Vickers hardness"],
}


TRAIN_PATH = "Data/train.csv"

def load_clean(path=TRAIN_PATH ):
    df = pd.read_csv(path)
    df = df.rename(columns={"σa": "sigma_a"})
    X = df.drop(columns=[c for c in DROP_FROM_X if c in df.columns])
    y = df["log10_Nf"]
    groups = df["group"] 
    folds = df["fold"] if "fold" in df.columns else None 
    return X, y, groups, folds


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    s = np.abs(y_pred - y_true)
    hits = s < np.log10(2)
    within_factor_2 = np.mean(hits)
    
    return {"rmse": rmse, "mae": mae, "r2": r2, "within_factor_2": within_factor_2}


def self_test():
    y_true = np.array([4.0, 5.0, 6.0])
    perfect = compute_metrics(y_true, y_true)
    assert perfect["rmse"] == 0.0 and perfect["r2"] == 1.0
    assert perfect["within_factor_2"] == 1.0
    missed = compute_metrics(y_true, np.array([4.0, 5.0, 6.31]))
    assert abs(missed["within_factor_2"] - 2 / 3) < 1e-9


def basquin_baseline(X_train, y_train, X_test):
    log_sigma_train = np.log10(X_train[["sigma_a"]]) 
    log_sigma_test = np.log10(X_test[["sigma_a"]])
    model = LinearRegression().fit(log_sigma_train, y_train)

    return model.predict(log_sigma_test)

def sklearn_fit_predict(model):
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

    results = []
    for f in sorted(folds.unique()):
        tr, te = folds != f, folds == f  
        y_pred = fit_predict(X[tr], y[tr], X[te])

        m = compute_metrics(y[te], y_pred)

        m["fold"] = f
        m["n_test"] = int(te.sum())
        results.append(m)
    return pd.DataFrame(results)


def feature_importance(model, X, y, folds, n_repeats=10, seed=0):

    blocked = [c for cols in COLLINEAR_BLOCKS.values() for c in cols]
    items = {c: [c] for c in X.columns if c not in blocked}
    items.update(COLLINEAR_BLOCKS)

    rng = np.random.default_rng(seed)
    scores = {name: [] for name in items}
 
    for f in sorted(folds.unique()):
        tr, te = folds != f, folds == f
        fitted = clone(model).fit(X[tr], y[tr])
        X_val = X[te].reset_index(drop=True)
        y_val = y[te].reset_index(drop=True)
        base = np.sqrt(np.mean((fitted.predict(X_val) - y_val) ** 2))
 
        for name, cols in items.items():
            for _ in range(n_repeats):
                X_perm = X_val.copy()
                order = rng.permutation(len(X_perm))
                X_perm[cols] = X_perm[cols].iloc[order].values
                rmse = np.sqrt(np.mean((fitted.predict(X_perm) - y_val) ** 2))
                scores[name].append(rmse - base)

    imp = pd.DataFrame({
        "mean": {k: np.mean(v) for k, v in scores.items()},
        "std": {k: np.std(v) for k, v in scores.items()},
    }).sort_values("mean", ascending=False)
    imp["is_block"] = [k in COLLINEAR_BLOCKS for k in imp.index]
    return imp


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

    return all_results.groupby("model")[metrics].agg(["mean", "std"])


PARAM_GRIDS = {
    "ridge": (
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": np.logspace(-3, 3, 13)},
    ),
    "random_forest": (
        RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        {
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3, 5],
            "max_features": [1.0, 0.5],
        },
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingRegressor(random_state=0),
        {
            "learning_rate": [0.03, 0.1, 0.3],
            "max_leaf_nodes": [15, 31],
            "min_samples_leaf": [10, 20, 40],
        },
    ),
}


def tune_all(X, y, folds):
    cv = PredefinedSplit(folds)
    tuned, rows = {}, []
    for name, (model, grid) in PARAM_GRIDS.items():
        search = GridSearchCV(model, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X, y)
        best = search.best_index_
        scores = -search.cv_results_["mean_test_score"]
        rows.append({
            "model": name,
            "rmse_mean": -search.best_score_,
            "rmse_std": search.cv_results_["std_test_score"][best],
            "grid_worst": scores.max(),
            "n_combos": len(scores),
            "best_params": search.best_params_,
        })
        tuned[name] = search.best_estimator_
    return tuned, pd.DataFrame(rows).set_index("model")


if __name__ == "__main__":
    
    self_test()

    X, y, groups, folds = load_clean()
    print(f"Training data: {TRAIN_PATH}")
    print(f"{len(X)} rows, {X.shape[1]} features, {groups.nunique()} material groups\n")

    default = run_all(X, y, groups, folds)
    print("5-fold grouped CV, default settings")
    print(default.round(3), "\n")

    tuned, summary = tune_all(X, y, folds)
    gain = pd.DataFrame({
        "default_rmse": default[("rmse", "mean")],
        "tuned_rmse": summary["rmse_mean"],
        "gain": default[("rmse", "mean")] - summary["rmse_mean"],
        "fold_std": summary["rmse_std"],
        "grid_worst": summary["grid_worst"],
    }).dropna()
    print("Tuning: gain vs spread between folds")
    print(gain.round(3), "\n")
    print("Best settings")
    print(summary["best_params"].to_string(), "\n")

    best_name = summary["rmse_mean"].idxmin()
    final_model = tuned[best_name]


    print(f"Feature importance for {best_name} (RMSE increase when shuffled)")
    print(feature_importance(final_model, X, y, folds).round(3).head(8), "\n")

    X_test, y_test, _, _ = load_clean("Data/test.csv")
    X_test = X_test[X.columns]
    preds = {"basquin_baseline": basquin_baseline(X, y, X_test)}
    for name, model in tuned.items():
        preds[name] = model.predict(X_test)
    test_results = pd.DataFrame({n: compute_metrics(y_test, p) for n, p in preds.items()}).T

    print(f"Test set ({len(X_test)} rows, {X_test.shape[1]} features), final model: {best_name}")
    print(test_results.round(3), "\n")

    sample = pd.DataFrame({
        "sigma_a": X_test["sigma_a"],
        "true_Nf": 10 ** y_test,
        "pred_Nf": 10 ** preds[best_name],
        "ratio": 10 ** (preds[best_name] - y_test),
    }).sample(10, random_state=0)
    print("Sample predictions on the test set")
    print(sample.round({"sigma_a": 0, "true_Nf": 0, "pred_Nf": 0, "ratio": 2}).to_string())
 
