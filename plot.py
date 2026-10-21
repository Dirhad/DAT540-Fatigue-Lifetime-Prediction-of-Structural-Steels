

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from evaluate import (load_clean, tune_all, feature_importance,
                      basquin_baseline)

OUT = "figures"
TRAIN = "Data/train.csv"
TEST = "Data/test.csv"
TEST_UNCENSORED = "Data/test_with_runouts.csv"   # only for the run-out share

BLUE = "#2a78d6"     # models
GREY = "#8c8a84"     # baseline, reference material
RED = "#b03030"      # the one thing the reader should notice
BAND = "#dfe8f5"     # factor-two band
INK = "#3f3e3c"

# Column names as they should read in a figure, not as they appear in the CSV.
PRETTY = {
    "sigma_a": r"$\sigma_a$",
    "Type of test": "Loading mode",
}
MODEL_NAMES = {
    "basquin_baseline": "Basquin baseline",
    "ridge": "Ridge",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "font.size": 9,
    "axes.edgecolor": "#c9c7c0",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path)


def decade(v, _):
    """Axis labels as 10^n instead of 1e+06."""
    return f"$10^{{{int(np.log10(v))}}}$"


def pretty(name):
    return PRETTY.get(name, name)


def repeat_test_scatter(df):
    """Spread between repeat tests at identical conditions, in log10 cycles.

    Same material state, same stress amplitude, same loading mode, different
    measured lives. The model sees identical inputs and must return one number,
    so this spread is error it is guaranteed to make: the noise floor.

    Slightly overestimated, because the preprocessing dropped exact duplicate
    rows, which removes replicate pairs that happened to share a rounded life.
    """
    cell = df.groupby(["group", "sigma_a", "Type of test"])["log10_Nf"]
    residual = (df["log10_Nf"] - cell.transform("mean"))[cell.transform("size") > 1]
    n_cells = int((cell.size() > 1).sum())
    return np.sqrt((residual ** 2).sum() / (len(residual) - n_cells))


def fig1_predicted_vs_measured(y_test, preds, best_name):
    """Does the model predict unseen material states, and does it beat Basquin?

    Two panels, not four. Ridge and random forest are in the results table; the
    contrast worth a figure is the baseline against the model being delivered,
    and four panels would halve the size of each.
    """
    shown = ["basquin_baseline", best_name]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)

    all_values = np.concatenate([np.asarray(y_test)] + [np.asarray(preds[k]) for k in shown])
    lo, hi = 10 ** all_values.min() / 1.6, 10 ** all_values.max() * 1.6
    corner = np.array([lo, hi])

    for ax, name in zip(axes, shown):
        y_pred = np.asarray(preds[name])
        ax.fill_between(corner, corner / 2, corner * 2, color=BAND, zorder=0)
        ax.plot(corner, corner, color=INK, linewidth=0.9, zorder=1)
        ax.scatter(10 ** y_test, 10 ** y_pred, s=11, alpha=0.55, zorder=2,
                   color=GREY if name == "basquin_baseline" else RED,
                   edgecolors="white", linewidths=0.3)

        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        within = np.mean(np.abs(y_pred - y_test) < np.log10(2))
        ax.text(0.04, 0.96,
                f"{MODEL_NAMES[name]}\nRMSE {rmse:.3f}\n{within:.0%} inside the band",
                transform=ax.transAxes, va="top", fontsize=8.5, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="#d8d6cf", linewidth=0.7))

        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
               xlabel="Measured cycles to failure")
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(FuncFormatter(decade))
        ax.yaxis.set_major_formatter(FuncFormatter(decade))
        ax.grid(color="#eceae4", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Predicted cycles to failure")
    axes[1].text(0.97, 0.05, "shaded: within a factor of 2", transform=axes[1].transAxes,
                 ha="right", fontsize=8, color="#5d7fae")
    fig.subplots_adjust(wspace=0.08)
    save(fig, "1_predicted_vs_measured")




def fig2_feature_importance(imp):
    """Which columns does the delivered model actually rely on?

    Only columns whose effect clears three times their own scatter. The rest
    sit at zero within noise, and listing them implies a ranking that the data
    does not support.
    """
    keep = imp.sort_values("mean").tail(6)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(keep) + 1.1))

    ax.barh(range(len(keep)), keep["mean"], xerr=keep["std"], height=0.62,
            color=[RED if block else BLUE for block in keep["is_block"]],
            error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 2})
    for row, (value, spread) in enumerate(zip(keep["mean"], keep["std"])):
        ax.text(value + spread + 0.012, row, f"{value:.2f}", va="center", fontsize=8.5)

    ax.set_yticks(range(len(keep)), [pretty(i) for i in keep.index])
    ax.set_xlim(0, keep["mean"].max() * 1.18)
    ax.set_xlabel("RMSE increase when the column is shuffled (log$_{10}$ cycles)")
    ax.text(0.99, 0.04, "red: three collinear strength columns, shuffled as one block",
            transform=ax.transAxes, ha="right", fontsize=8, color=RED)
    ax.grid(axis="x", color="#eceae4", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "2_feature_importance")
    return keep



def fig3_sn_curves(train, n_states=5):
    """Why does Basquin fail across materials when it holds within one?

    One line per material state, plus the pooled fit the baseline actually
    uses. The individual curves are near-parallel but sit at different levels,
    so a single line through all of them cannot follow any of them.

    States are picked deterministically: those with at least 12 specimens
    spanning more than 100 MPa, then spread evenly across tensile strength so
    the vertical offset between curves is visible rather than accidental.
    """
    stats = train.groupby("group").agg(n=("sigma_a", "size"),
                                       tensile=("Tensile strength", "first"),
                                       span=("sigma_a", lambda s: s.max() - s.min()))
    usable = stats[(stats["n"] >= 12) & (stats["span"] > 100)].sort_values("tensile")
    picked = usable.index[np.linspace(0, len(usable) - 1, n_states).astype(int)]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    shades = plt.cm.viridis(np.linspace(0.08, 0.88, len(picked)))

    for colour, state in zip(shades, picked):
        rows = train[train["group"] == state]
        ax.scatter(10 ** rows["log10_Nf"], rows["sigma_a"], s=16, color=colour,
                   alpha=0.75, edgecolors="white", linewidths=0.4, zorder=3)
        # Basquin within one state: a straight line in log-log.
        slope, intercept = np.polyfit(np.log10(rows["sigma_a"]), rows["log10_Nf"], 1)
        grid = np.linspace(rows["sigma_a"].min(), rows["sigma_a"].max(), 50)
        ax.plot(10 ** (intercept + slope * np.log10(grid)), grid, color=colour,
                linewidth=1.5, zorder=2,
                label=f"{stats.loc[state, 'tensile']:.0f} MPa tensile")

    # The single line the baseline fits through every material at once.
    slope, intercept = np.polyfit(np.log10(train["sigma_a"]), train["log10_Nf"], 1)
    grid = np.linspace(train["sigma_a"].min(), train["sigma_a"].max(), 50)
    ax.plot(10 ** (intercept + slope * np.log10(grid)), grid, color=RED,
            linewidth=2, linestyle="--", zorder=4, label="Pooled fit (baseline)")

    ax.set(xscale="log", yscale="log",
           xlabel="Cycles to failure", ylabel=r"Stress amplitude $\sigma_a$ (MPa)")
    ax.xaxis.set_major_formatter(FuncFormatter(decade))
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([200, 300, 400, 600, 900])
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(color="#eceae4", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "3_sn_curves")

    within = train.groupby("group").apply(
        lambda d: np.corrcoef(np.log10(d["sigma_a"]), d["log10_Nf"])[0, 1]
        if len(d) > 3 else np.nan, include_groups=False)
    return within.median()

def main():
    X, y, groups, folds = load_clean(TRAIN)
    X_test, y_test, _, _ = load_clean(TEST)
    X_test = X_test[X.columns]          # same column order as training
    train = pd.read_csv(TRAIN).rename(columns={"σa": "sigma_a"})
 
    print("Tuning (this is the slow part)...")
    tuned, summary = tune_all(X, y, folds)
    best_name = summary["rmse_mean"].idxmin()
    final_model = tuned[best_name]
 
    # Only the two models figure 1 shows.
    preds = {"basquin_baseline": basquin_baseline(X, y, X_test),
             best_name: final_model.predict(X_test)}
 
    floor = repeat_test_scatter(train)
    # 30 repeats, not 10: with 10 the spread was large enough that the ranking
    # of the smaller features moved between runs.
    importance = feature_importance(final_model, X, y, folds, n_repeats=30)
 
    print("\nFigures")
    fig1_predicted_vs_measured(y_test, preds, best_name)
    kept = fig2_feature_importance(importance)
    within_state = fig3_sn_curves(train)
 
    # Everything a caption or a sentence in the report needs, in one place.
    cv_best = summary.loc[best_name, "rmse_mean"]
    print(f"\nFinal model: {best_name}")
    print(f"Repeat-test scatter (noise floor): {floor:.3f}")
    print(f"Best model CV RMSE: {cv_best:.3f}  ->  {cv_best - floor:.3f} above the floor")
 
    print("\nFig 1, test set")
    for name in ["basquin_baseline", best_name]:
        error = preds[name] - y_test
        rmse = np.sqrt(np.mean(error ** 2))
        print(f"  {MODEL_NAMES[name]:18s} RMSE {rmse:.3f}"
              f"  within factor 2 {np.mean(np.abs(error) < np.log10(2)):.1%}"
              f"  typical miss factor {10 ** rmse:.1f}")
 
    print("\nFig 2, feature importance")
    print(kept.sort_values("mean", ascending=False)[["mean", "std"]].round(3).to_string())
 
    print(f"\nFig 3, median within-state correlation "
          f"log(sigma_a) vs log(Nf): {within_state:.2f}")

if __name__ == "__main__":
    main()