import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from evaluate import load_clean, compute_metrics

#---------------------------------------------------------------------------------
# How much does a random split by row flatter the score?
# Same model and settings as the final model in evaluate.py, same failed tests,
# only the way the data is split changes.
# Random split: tests from one material state can end up in both train and test.
# Group split: every material state is on one side only, as in the main analysis.
#---------------------------------------------------------------------------------

N_SEEDS = 10
TEST_SIZE = 0.2

# Best settings for gradient boosting from the grid search in evaluate.py
BEST = {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 10}


def all_failed_tests():
    # Train and test from data_prep.py put back together (run-outs already removed)
    X_tr, y_tr, g_tr, _ = load_clean("Data/train.csv")
    X_te, y_te, g_te, _ = load_clean("Data/test.csv")
    X = pd.concat([X_tr, X_te[X_tr.columns]], ignore_index=True)
    y = pd.concat([y_tr, y_te], ignore_index=True)
    groups = pd.concat([g_tr, g_te], ignore_index=True)
    return X, y, groups


def score_split(splitter, X, y, groups):
    # Fit on the train part, score once on the test part, for every seed
    rows = []
    for seed in range(N_SEEDS):
        splitter.random_state = seed
        tr, te = next(splitter.split(X, y, groups))
        model = HistGradientBoostingRegressor(random_state=0, **BEST).fit(X.iloc[tr], y.iloc[tr])
        rows.append(compute_metrics(y.iloc[te], model.predict(X.iloc[te])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    X, y, groups = all_failed_tests()
    print(f"{len(X)} failed tests, {groups.nunique()} material states, {N_SEEDS} splits each\n")

    results = {
        "random split by row": score_split(ShuffleSplit(n_splits=1, test_size=TEST_SIZE), X, y, groups),
        "split by material state": score_split(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE), X, y, groups),
    }

    summary = pd.DataFrame({
        name: {f"{m}_{s}": getattr(res[m], s)() for m in ["rmse", "r2", "within_factor_2"] for s in ["mean", "std"]}
        for name, res in results.items()
    }).T
    print(summary.round(3))
