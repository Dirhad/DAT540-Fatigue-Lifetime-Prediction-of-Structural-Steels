# Makes the figures for the report. Saves them as PNG in figures/
# Run evaluate.py first if you want the numbers, this file only draws.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluate import load_clean, tune_all, basquin_baseline, cross_validate_folds


OUT = "figures"

# Fixed colors, one per steel grade (same grade = same color)
GRADE_COLORS = {
    "S25C": "#2a78d6",
    "S35C": "#eb6834",
    "S40C": "#1baf7a",
    "S45C": "#eda100",
    "S55C": "#e87ba4",
}
MAIN = "#2a78d6" # Color for single-series plots
GREY = "#898781" # Baseline, reference lines
INK = "#52514e"  # Text and axes

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", f"{OUT}/{name}.png")


def replicate_scatter(df):
    # Same steel, same stress, same test type, but still different lifetimes. That spread is noise
    # no model can explain, so it is the lowest RMSE we can hope for
    cell = df.groupby(["group", "sigma_a", "Type of test"])["log10_Nf"]
    size = cell.transform("size")
    resid = (df["log10_Nf"] - cell.transform("mean"))[size > 1]
    n_cells = (cell.size() > 1).sum()
    return np.sqrt((resid ** 2).sum() / (len(resid) - n_cells)) 


def plot_target_distribution(raw):
    # Why the target is log10(Nf) and why run-outs are left out. Uses the raw Excel file.
    nf = raw["Nf"]
    runout = nf >= 1e7
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))

    left.hist(nf[~runout], bins=40, color=MAIN)
    left.set_xlabel("Cycles to failure, Nf")
    left.set_ylabel("Number of tests")
    left.set_title(f"Failed tests, raw Nf (skew {nf[~runout].skew():.1f})", loc="left", fontsize=10)

    bins = np.linspace(4, 10, 49)
    right.hist(np.log10(nf[~runout]), bins=bins, color=MAIN, label=f"Failed ({(~runout).sum()})")
    right.hist(np.log10(nf[runout]), bins=bins, color=GREY, label=f"Run-out, Nf ≥ 10⁷ ({runout.sum()})")
    right.axvline(7, color=INK, linestyle="--", linewidth=1)
    right.set_xlabel("log10(Nf)")
    right.set_title(f"All tests, log scale (skew of failed {np.log10(nf[~runout]).skew():.1f})", loc="left", fontsize=10)
    right.legend(frameon=False, fontsize=8)
    for ax in (left, right):
        ax.grid(axis="x", visible=False)
    save(fig, "1_target_distribution")


def plot_sn_curves(df):
    # S-N data, raw stress and stress divided by tensile strength. 
    grade = df[[f"mat_{g}" for g in GRADE_COLORS]].idxmax(axis=1).str[4:]
    ratio = df["sigma_a"] / df["Tensile strength"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, y, label in [(axes[0], df["sigma_a"], "Stress amplitude σa (MPa)"),
                         (axes[1], ratio, "σa / tensile strength")]:
        for g, color in GRADE_COLORS.items():
            rows = grade == g
            ax.scatter(10 ** df.loc[rows, "log10_Nf"], y[rows], s=10, color=color, alpha=0.7,
                       edgecolors="white", linewidths=0.4, label=f"{g} ({rows.sum()})")
        r = np.corrcoef(y, df["log10_Nf"])[0, 1]
        ax.set_title(f"{label.split(' (')[0]}, r = {r:.2f} with log10(Nf)", loc="left", fontsize=10)
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.set_xlabel("Cycles to failure, Nf (log scale)")
    axes[1].legend(title="Grade (rows)", frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("S-N data per steel grade (train set)", x=0.02, ha="left")
    save(fig, "2_sn_curves")


def plot_correlation(df):
    # Correlation between the numeric inputs and the target
    cols = ["sigma_a", "0.2% proof stress", "Tensile strength", "Vickers hardness", "Elongation",
            "Reduction of area", "Tempering temp", "Quenching temp", "Normalizing temp",
            "Reduction Ratio", "dA", "dB", "dC", "log10_Nf"]
    corr = df[cols].corr()
    corr = corr.where(np.tril(np.ones(corr.shape, dtype=bool)))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(i + 1):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Correlation between features and log10(Nf) (train set)", loc="left")
    save(fig, "3_correlation")


def plot_pred_vs_true(y_test, preds, best_name):
    # Predicted vs true on the test set, one panel per model.
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    all_values = np.concatenate([y_test] + list(preds.values()))
    lo, hi = 10 ** all_values.min() / 1.5, 10 ** all_values.max() * 1.5
    line = np.array([lo, hi])
    for ax, (name, y_pred) in zip(axes.flat, preds.items()):
        ax.plot(line, line, color=INK, linewidth=1, label="Perfect prediction")
        ax.plot(line, line * 2, color=GREY, linewidth=1, linestyle="--", label="Factor 2")
        ax.plot(line, line / 2, color=GREY, linewidth=1, linestyle="--")
        color = GREY if name == "basquin_baseline" else MAIN
        ax.scatter(10 ** y_test, 10 ** y_pred, s=12, color=color, alpha=0.6, edgecolors="white", linewidths=0.4)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        within = np.mean(np.abs(y_pred - y_test) < np.log10(2))
        ax.set_title(f"{name}\nRMSE {rmse:.3f}, {within:.0%} within factor 2", loc="left", fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("True cycles to failure")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted cycles to failure")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Predicted vs true lifetime on the test set", x=0.02, ha="left")
    save(fig, "4_pred_vs_true")


def plot_model_comparison(results, floor):
    # Cross validation error per model, with the experimental scatter as a floor
    fig, ax = plt.subplots(figsize=(7, 3.5))
    results = results.sort_values("rmse_mean", ascending=False)
    colors = [GREY if m == "basquin_baseline" else MAIN for m in results.index]
    ax.barh(results.index, results["rmse_mean"], xerr=results["rmse_std"], color=colors, height=0.6,
            error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    for i, v in enumerate(results["rmse_mean"]):
        ax.text(v + results["rmse_std"].iloc[i] + 0.01, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.axvline(floor, color=INK, linestyle="--", linewidth=1,
               label=f"Scatter between repeat tests ≈ {floor:.2f} (best possible)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9)
    ax.set_xlim(0, results["rmse_mean"].max() + 0.12)
    ax.set_xlabel("RMSE, 5-fold grouped CV (log10 cycles, lower is better)")
    ax.set_title("Model comparison (tuned settings, error bars = std over folds)", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "5_model_comparison")


if __name__ == "__main__":
    df = pd.read_csv("Data/train.csv").rename(columns={"σa": "sigma_a"})
    X, y, groups, folds = load_clean()
    X_test, y_test, _, _ = load_clean("Data/test.csv")
    X_test = X_test[X.columns]

    # Same steps as evaluate.py, tune on train, pick lowest CV error, predict test once
    tuned, summary = tune_all(X, y, folds)
    best_name = summary["rmse_mean"].idxmin()

    # Test predictions for every model. 
    preds = {"basquin_baseline": basquin_baseline(X, y, X_test)}
    for name, model in tuned.items():
        preds[name] = model.predict(X_test)

    basquin_cv = cross_validate_folds(basquin_baseline, X, y, folds)
    results = summary[["rmse_mean", "rmse_std"]].copy()
    results.loc["basquin_baseline"] = [basquin_cv["rmse"].mean(), basquin_cv["rmse"].std()]

    floor = replicate_scatter(df)
    print(f"Scatter between repeat tests: {floor:.3f}")

    # Data exploration
    raw = pd.read_excel("Dataset/NIMS_FATIGUE.xlsx")
    raw.columns = raw.columns.str.strip()
    plot_target_distribution(raw)
    plot_sn_curves(df)
    plot_correlation(df)

    # Results
    plot_pred_vs_true(y_test, preds, best_name)
    plot_model_comparison(results, floor)

