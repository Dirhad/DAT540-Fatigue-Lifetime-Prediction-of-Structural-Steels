# Builds the four report figures as PNG in figures/.
#
# Figures carry no titles. In the report the caption below each figure states
# the finding, and a title inside the image would only repeat it and eat
# vertical space on a four-page limit.

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from evaluate import (load_clean, tune_all, feature_importance,
                      basquin_baseline, cross_validate_folds, sklearn_fit_predict)

warnings.filterwarnings("ignore", category=UserWarning)

OUT = "figures"

MAIN = "#2a78d6"     # models
GREY = "#8c8a84"     # baseline, reference lines
ACCENT = "#b03030"   # the one thing the reader should notice
BAND = "#dfe8f5"     # factor-two band
INK = "#3f3e3c"

LABELS = {
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
    "axes.labelsize": 9,
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
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", f"{OUT}/{name}.png")


def sci(v, _):
    """Axis labels as 10^n rather than 1e+06."""
    return f"$10^{{{int(np.log10(v))}}}$"


def replicate_scatter(df):
    """Spread between repeat tests at identical conditions.

    Same material, same stress, same test type, different measured lives. No
    model can explain that, so it is the lowest RMSE anything could reach.
    """
    cell = df.groupby(["group", "sigma_a", "Type of test"])["log10_Nf"]
    size = cell.transform("size")
    resid = (df["log10_Nf"] - cell.transform("mean"))[size > 1]
    n_cells = (cell.size() > 1).sum()
    return np.sqrt((resid ** 2).sum() / (len(resid) - n_cells))


# --------------------------------------------------------------------------
# 1. Baseline against the final model on the held-out test set
# --------------------------------------------------------------------------
def plot_pred_vs_true(y_test, preds, best_name):
    """Two panels only. Ridge and random forest live in the results table;
    putting all four here would halve the size of the contrast that matters."""
    shown = ["basquin_baseline", best_name]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)

    values = np.concatenate([np.asarray(y_test)] + [np.asarray(preds[k]) for k in shown])
    lo, hi = 10 ** values.min() / 1.6, 10 ** values.max() * 1.6
    edge = np.array([lo, hi])

    for ax, name in zip(axes, shown):
        y_pred = np.asarray(preds[name])
        ax.fill_between(edge, edge / 2, edge * 2, color=BAND, zorder=0)
        ax.plot(edge, edge, color=INK, linewidth=0.9, zorder=1)

        colour = GREY if name == "basquin_baseline" else ACCENT
        ax.scatter(10 ** y_test, 10 ** y_pred, s=11, color=colour, alpha=0.55,
                   edgecolors="white", linewidths=0.3, zorder=2)

        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        within = np.mean(np.abs(y_pred - y_test) < np.log10(2))
        ax.text(0.04, 0.96, f"{LABELS[name]}\nRMSE {rmse:.3f}\n{within:.0%} inside the band",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="#d8d6cf", linewidth=0.7))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(FuncFormatter(sci))
        ax.yaxis.set_major_formatter(FuncFormatter(sci))
        ax.set_xlabel("Measured cycles to failure")
        ax.grid(color="#eceae4", linewidth=0.6, zorder=0)

    axes[0].set_ylabel("Predicted cycles to failure")
    # One shared explanation of the band, placed where it cannot cover points.
    axes[1].text(0.97, 0.05, "shaded: within a factor of 2", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=8, color="#5d7fae")
    fig.subplots_adjust(wspace=0.08)
    save(fig, "1_pred_vs_true")


# --------------------------------------------------------------------------
# 2. Cross-validated error per model, every fold shown
# --------------------------------------------------------------------------
def plot_model_comparison(per_fold, floor):
    """Dots and whiskers, not bars. With five folds the individual values are
    the evidence; a bar from zero hides them and wastes the axis on a range no
    model occupies."""
    order = per_fold.groupby("model")["rmse"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(7.2, 2.9))

    ax.axvspan(0, floor, color="#f2e3e3", zorder=0)
    ax.axvline(floor, color=ACCENT, linewidth=1.1, zorder=1)

    rng = np.random.default_rng(0)
    for i, name in enumerate(order):
        vals = per_fold.loc[per_fold["model"] == name, "rmse"].to_numpy()
        colour = GREY if name == "basquin_baseline" else MAIN
        ax.scatter(vals, np.full_like(vals, i) + rng.uniform(-0.09, 0.09, len(vals)),
                   s=18, color=colour, alpha=0.5, edgecolors="white",
                   linewidths=0.4, zorder=2)
        ax.plot([vals.min(), vals.max()], [i, i], color=colour, linewidth=1, zorder=2)
        ax.scatter([vals.mean()], [i], s=70, marker="|", color=colour,
                   linewidths=2.2, zorder=3)
        ax.text(vals.max() + 0.012, i, f"{vals.mean():.3f}", va="center",
                fontsize=8.5, color=INK)

    ax.set_yticks(range(len(order)), [LABELS[m] for m in order])
    ax.set_xlim(0.2, per_fold["rmse"].max() + 0.07)
    ax.set_xlabel("RMSE per fold, 5-fold grouped CV (log$_{10}$ cycles)")
    ax.text(floor - 0.008, -0.42, f"repeat-test scatter {floor:.2f}",
            ha="right", va="center", fontsize=8, color=ACCENT)
    ax.grid(axis="x", color="#eceae4", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "2_model_comparison")


# --------------------------------------------------------------------------
# 3. Permutation importance for the final model
# --------------------------------------------------------------------------
def plot_feature_importance(imp):
    """Only columns whose effect exceeds their own scatter. The rest sit at
    zero within noise and listing them suggests a ranking that is not there."""
    keep = imp[imp["mean"] > 3 * imp["std"]].sort_values("mean")
    keep = keep[keep["mean"] >= 0.01]
    keep.index = [r"$\sigma_a$" if i == "sigma_a" else i for i in keep.index]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(keep) + 1.1))

    colours = [ACCENT if b else MAIN for b in keep["is_block"]]
    ax.barh(range(len(keep)), keep["mean"], xerr=keep["std"], color=colours,
            height=0.62, error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 2})
    for i, v in enumerate(keep["mean"]):
        ax.text(v + keep["std"].iloc[i] + 0.012, i, f"{v:.2f}", va="center",
                fontsize=8.5, color=INK)

    ax.set_yticks(range(len(keep)), keep.index)
    ax.set_xlim(0, keep["mean"].max() * 1.18)
    ax.set_xlabel("RMSE increase when the column is shuffled (log$_{10}$ cycles)")
    ax.text(0.99, 0.04, "red: three collinear strength columns, shuffled as one",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=ACCENT)
    ax.grid(axis="x", color="#eceae4", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "3_feature_importance")


# --------------------------------------------------------------------------
# 4. Residuals against stress, with the censoring that explains them
# --------------------------------------------------------------------------
def plot_residuals(y_test, preds, best_name, sigma_test, runout_file):
    """The censoring strip sits directly above the residuals, sharing the x
    axis. A twin y axis was the other option, but its line crossed the point
    cloud and had to be read against a scale on the far side of the figure."""
    resid = preds[best_name] - y_test
    bins = pd.qcut(sigma_test, 6)
    trend = resid.groupby(bins, observed=True).mean()
    centres = [iv.mid for iv in trend.index]
    widths = [iv.right - iv.left for iv in trend.index]

    # Same bin edges on the uncensored file, so the two panels line up.
    edges = [iv.left for iv in trend.index] + [trend.index[-1].right]
    full = pd.read_csv(runout_file).rename(columns={"σa": "sigma_a"})
    share = full.groupby(pd.cut(full["sigma_a"], edges, include_lowest=True),
                         observed=True)["runout"].mean()

    fig, (top, ax) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 4.2],
                                               "hspace": 0.08})

    top.bar(centres, share.values, width=[w * 0.92 for w in widths],
            color="#cdc9c0", edgecolor="white", linewidth=0.8)
    for c, v in zip(centres, share.values):
        top.text(c, v + 0.03, f"{v:.0%}", ha="center", fontsize=7.5, color=INK)
    top.set_ylim(0, 0.68)
    top.set_ylabel("run-outs\nremoved", fontsize=8, linespacing=1.4)
    top.set_yticks([])
    top.spines["left"].set_visible(False)
    top.grid(False)

    ax.axhspan(-np.log10(2), np.log10(2), color=BAND, zorder=0)
    ax.axhline(0, color=INK, linewidth=0.9, zorder=1)

    inside = np.abs(resid) < np.log10(2)
    ax.scatter(sigma_test[inside], resid[inside], s=13, color=MAIN, alpha=0.5,
               edgecolors="white", linewidths=0.3, zorder=2)
    ax.scatter(sigma_test[~inside], resid[~inside], s=13, color=GREY, alpha=0.55,
               edgecolors="white", linewidths=0.3, zorder=2)
    ax.plot(centres, trend.values, color=ACCENT, linewidth=1.9, marker="o",
            markersize=5, zorder=4, label="Mean residual per stress bin")

    ax.set_xlabel("Stress amplitude $\\sigma_a$ (MPa)")
    ax.set_ylabel("log$_{10}$(predicted) \u2212 log$_{10}$(measured)")
    ax.text(0.014, 0.97, "overpredicted", transform=ax.transAxes, va="top",
            fontsize=8, color="#5d7fae")
    ax.text(0.014, 0.03, "underpredicted", transform=ax.transAxes, va="bottom",
            fontsize=8, color="#5d7fae")
    ax.text(0.986, 0.97, "shaded: within a factor of 2", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#5d7fae")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(color="#eceae4", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    save(fig, "4_residuals_vs_stress")
    return pd.DataFrame({"mean_residual": trend.values,
                         "runout_share": share.values}, index=trend.index)


if __name__ == "__main__":
    X, y, groups, folds = load_clean()
    X_test, y_test, _, _ = load_clean("Data/test.csv")
    X_test = X_test[X.columns]

    tuned, summary = tune_all(X, y, folds)
    best_name = summary["rmse_mean"].idxmin()
    final_model = tuned[best_name]

    preds = {"basquin_baseline": basquin_baseline(X, y, X_test)}
    for name, model in tuned.items():
        preds[name] = model.predict(X_test)

    # Per-fold CV error for every model, tuned settings, so figure 2 can show
    # the individual folds rather than a mean and a bar.
    per_fold = []
    for name, model in {"basquin_baseline": None, **tuned}.items():
        fp = basquin_baseline if model is None else sklearn_fit_predict(model)
        res = cross_validate_folds(fp, X, y, folds)
        res["model"] = name
        per_fold.append(res)
    per_fold = pd.concat(per_fold, ignore_index=True)

    train = pd.read_csv("Data/train.csv").rename(columns={"σa": "sigma_a"})
    floor = replicate_scatter(train)
    print(f"Repeat-test scatter: {floor:.3f}")
    print(f"Final model: {best_name}\n")

    plot_pred_vs_true(y_test, preds, best_name)
    plot_model_comparison(per_fold, floor)
    plot_feature_importance(feature_importance(final_model, X, y, folds))
    table = plot_residuals(y_test, preds, best_name, X_test["sigma_a"],
                           "Data/test_with_runouts.csv")

    print("\nResidual bias against censoring, per stress bin")
    print(table.round(3).to_string())
    