import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

INCLUDE_RUNOUTS = False
RUNOUT_LIMIT = 1e7
TEST_SIZE = 0.2
N_FOLDS = 5
SEED = 42

TEMP_COLS = ["Normalizing temp", "Quenching temp", "Tempering temp"]
DROP_COLS = ["DS No. PDF", "Heat", "Upper yield stress", "True fracture stress"]

steps = []

df = pd.read_excel("Dataset/NIMS_FATIGUE.xlsx")
df.columns = df.columns.str.strip()
steps.append(("loaded", len(df)))

is_duplicate = df.duplicated() & (df["Nf"] < RUNOUT_LIMIT)
df = df[~is_duplicate].reset_index(drop=True)
steps.append(("duplicates removed (failed tests only)", len(df)))

impossible = df["0.2% proof stress"] > df["Tensile strength"]
df = df[~impossible].reset_index(drop=True)
steps.append(("proof stress > tensile strength removed", len(df)))

df = df.drop(columns=DROP_COLS)

state_cols = [c for c in df.columns if c not in ("Nf", "σa", "Type of test")]
df["group"] = df.groupby(state_cols, dropna=False).ngroup()
steps.append((f"groups made ({df['group'].nunique()} material states)", len(df)))

df["runout"] = (df["Nf"] >= RUNOUT_LIMIT).astype(int)
df["log10_Nf"] = np.log10(df["Nf"])
df = df.drop(columns=["Nf"])

df["Type of test"] = (df["Type of test"] == "Uniaxial loading").astype(int)
df = pd.get_dummies(df, columns=["Testing Material"], prefix="mat", dtype=int)

splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(splitter.split(df, groups=df["group"]))
train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()
steps.append(("split: train rows", len(train)))
steps.append(("split: test rows", len(test)))

if not INCLUDE_RUNOUTS:
    train = train[train["runout"] == 0].copy()
    test = test[test["runout"] == 0].copy()
steps.append(("after run-out choice: train rows", len(train)))
steps.append(("after run-out choice: test rows", len(test)))

mat_cols = [c for c in train.columns if c.startswith("mat_")]
grade = train[mat_cols].idxmax(axis=1)
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train["fold"] = -1
for k, (_, val_idx) in enumerate(cv.split(train, grade, groups=train["group"])):
    train.iloc[val_idx, train.columns.get_loc("fold")] = k
test["fold"] = -1

for col in TEMP_COLS:
    train[col + " missing"] = train[col].isna().astype(int)
    test[col + " missing"] = test[col].isna().astype(int)
    median = train[col].median()
    train[col] = train[col].fillna(median)
    test[col] = test[col].fillna(median)

same_flag = "Tempering temp missing"
assert (train[same_flag] == train["Quenching temp missing"]).all()
assert (test[same_flag] == test["Quenching temp missing"]).all()
train = train.drop(columns=same_flag)
test = test.drop(columns=same_flag)

assert train.isna().sum().sum() == 0 and test.isna().sum().sum() == 0, "missing values left"
assert not set(train["group"]) & set(test["group"]), "a group is in both train and test"
assert train.select_dtypes(exclude="number").empty, "non-numeric column left"
assert train.groupby("group")["fold"].nunique().max() == 1, "a group is split over folds"
assert sorted(train["fold"].unique()) == list(range(N_FOLDS)), "empty fold"

print("Step log (rows after each step)")
for name, n in steps:
    print(f"  {name}: {n}")
print("\nMaterial counts in train / test")
print(pd.DataFrame({"train": train[mat_cols].sum(), "test": test[mat_cols].sum()}))
print("\nRows and groups per fold")
print(train.groupby("fold").agg(rows=("group", "size"), groups=("group", "nunique")))
print(train.groupby("fold")[mat_cols].sum())
print("\nType of test (1 = uniaxial): train / test mean")
print(train["Type of test"].mean().round(3), test["Type of test"].mean().round(3))
print("\nlog10_Nf describe")
print(pd.DataFrame({"train": train["log10_Nf"].describe(), "test": test["log10_Nf"].describe()}))

os.makedirs("Data", exist_ok=True)
suffix = "_with_runouts" if INCLUDE_RUNOUTS else ""
train.to_csv(f"Data/train{suffix}.csv", index=False)
test.to_csv(f"Data/test{suffix}.csv", index=False)
print(f"\nSaved Data/train{suffix}.csv and Data/test{suffix}.csv")
