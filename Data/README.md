# Data files

Made by `data_prep.py` (run it from the Fatigue folder).

- `train.csv`, `test.csv`: run-outs (Nf >= 1e7) left out. Use these for the main analysis.
- `train_with_runouts.csv`, `test_with_runouts.csv`: same split, run-outs included. For the sensitivity check.

Target: `log10_Nf`.

Not features, drop them before fitting: `log10_Nf`, `group`, `runout`, `fold`.

- `group`: id of the material state. All rows of a group must stay on the same side of a split.
- `fold`: 0 to 4 in the train files, use it for grouped cross-validation. It is -1 in the test files.
- `runout`: 1 if Nf >= 1e7.

The temperature columns only take a few values per steel grade, so they are almost the same information as the `mat_` columns. Linear regression may be unstable with all of them, use Ridge or drop some.
