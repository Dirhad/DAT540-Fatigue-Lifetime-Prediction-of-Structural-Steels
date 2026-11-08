# DAT540 - Fatigue Lifetime Prediction of Structural Steels

Group 12, DAT540 Project 3. We predict the fatigue life (log10 of cycles to failure) of carbon steels from the NIMS Fatigue Data Sheets. The models are tested on material states they did not see during training.

GitHub repository: https://github.com/Dirhad/DAT540-Fatigue-Lifetime-Prediction-of-Structural-Steels

## Where to find things

| Path | Contents |
|---|---|
| `notebook.ipynb` | The full project from data review to conclusion, with all outputs. Start here. |
| `data_prep.py` | Cleaning, the split by material state and the cross-validation folds. Writes the files in `Data/`. |
| `evaluate.py` | Models, cross-validation, tuning, feature importance and the final test. |
| `plot.py` | Result figures. |
| `random_split.py` | Compares a split by row with a split by material state (Table 3 in the report). |
| `Dataset/NIMS_FATIGUE.xlsx` | Raw data. |
| `Data/` | Cleaned train and test sets, described in `Data/README.md`. |
| `figures/` | Saved figures. |
| `requirements.txt` | Python packages. |

## How to run

Run all commands from this folder. We used Python 3.12.

```
pip install -r requirements.txt
```

To run the whole project, open the notebook and run all cells from top to bottom:

```
pip install notebook
jupyter notebook notebook.ipynb
```

The scripts can also be run on their own, in this order:

```
python data_prep.py
python evaluate.py
python plot.py
python random_split.py
```
