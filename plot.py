import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from evaluate import load_clean, tune_all, feature_importance, basquin_baseline

OUT = "figures"
TRAIN = "Data/train.csv"
TEST = "Data/test.csv"

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
    # Label for Axis to show 10^n instead of 1e+06
    return f"$10^{{{int(np.log10(v))}}}$"


def pretty(name):
    # Look up the display name, or keep the column name if it has none
    return PRETTY.get(name, name)


def repeat_test_scatter(df):
    # Check for how much lifetimes differ when the same test is repeated
    # (same material, same stress, same loading mode)

    cell = df.groupby(["group", "sigma_a", "Type of test"])["log10_Nf"]
    residual = (df["log10_Nf"] - cell.transform("mean"))[cell.transform("size") > 1]
    n_cells = int((cell.size() > 1).sum())
    return np.sqrt((residual ** 2).sum() / (len(residual) - n_cells))


def basquin_line(stress, log_nf):
    # Straight line representation of a material's stress-life 
    b, a = np.polyfit(np.log10(stress), log_nf, 1)
    return lambda s: a + b * np.log10(s)


def fig1_predicted_vs_measured(y_test, preds, best_name):
    # Figure 1: Predicted vs measured life on the test.
    # Baseline (left) and best model (right)
    # A perfect model puts every point on the diagonal line

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


def fig2_feature_importance(imp, top=6):
    # Figure 2: Permutation importance for the best model
    # X axis is how much worse the RMSE gets when the column is shuffled

    keep = imp.sort_values("mean").tail(top) # The top 6
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


def fig3_sn_curves(df, n_drawn=6):
    # Figure 3: S-N Curves (stress vs cycles) for a few material states + Basquin error numbers

    # Basquin line per material state and loading mode, fitted on its own data
    residuals, n_states = [], 0
    for _, fit in df.groupby(["group", "Type of test"]):
        if len(fit) < 3 or fit["sigma_a"].nunique() < 2:
            continue
        line = basquin_line(fit["sigma_a"], fit["log10_Nf"])
        residuals.append(fit["log10_Nf"] - line(fit["sigma_a"]))
        n_states += 1
    residuals = pd.concat(residuals)

    # One line through all data using stress only
    raw = basquin_line(df["sigma_a"], df["log10_Nf"])

    # One line through all data using stress divided by tensile strength
    ratio = df["sigma_a"] / df["Tensile strength"]
    norm = basquin_line(ratio, df["log10_Nf"])

    # RMSE of each of the three
    within_state = {
        "per_state": np.sqrt((residuals ** 2).sum() / (len(residuals) - 2 * n_states)),
        "n_states": n_states,
        "global_raw": np.sqrt(np.mean((df["log10_Nf"] - raw(df["sigma_a"])) ** 2)),
        "global_norm": np.sqrt(np.mean((df["log10_Nf"] - norm(ratio)) ** 2)),
    }


    mode = df["Type of test"].value_counts().idxmax()
    mode_name = "Uniaxial loading" if mode == 1 else "Rotating bending"  # data_prep: 1 = uniaxial
    in_mode = df[df["Type of test"] == mode]
    candidates = [(cell["Tensile strength"].iloc[0], cell) for _, cell in in_mode.groupby("group")
                  if len(cell) >= 5 and cell["sigma_a"].nunique() >= 3]
    candidates.sort(key=lambda c: c[0])  # spread the picks over strength
    pick = np.unique(np.linspace(0, len(candidates) - 1, min(n_drawn, len(candidates))).round().astype(int))
    shown = [candidates[i] for i in pick]
    colors = plt.cm.Blues(np.linspace(0.5, 0.95, len(shown)))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ends, rows = [], []
    for (uts, fit), color in zip(shown, colors):
        line = basquin_line(fit["sigma_a"], fit["log10_Nf"])
        s = np.linspace(fit["sigma_a"].min(), fit["sigma_a"].max(), 100)

        ax.scatter(10 ** fit["log10_Nf"], fit["sigma_a"], s=10, color=color, alpha=0.35,
                   edgecolors="none", zorder=2)
        ax.plot(10 ** line(s), s, color=color, linewidth=1.8, zorder=3)

        ends.append([10 ** line(s[0]), s[0], f"UTS {uts:.0f} MPa", color])
        b = 1 / np.polyfit(np.log10(fit["sigma_a"]), fit["log10_Nf"], 1)[0]
        rows.append({"UTS (MPa)": uts, "n": len(fit), "b": round(b, 3)})


    lives = 10 ** in_mode["log10_Nf"]
    top = in_mode["sigma_a"].max() * 1.1
    gap = 0.055 * top
    label_x = lives.max() * 2.5
    ends.sort(key=lambda e: e[1])
    label_y = [e[1] for e in ends]
    for i in range(1, len(label_y)):
        label_y[i] = max(label_y[i], label_y[i - 1] + gap)
    for (x, y_end, text, color), y_label in zip(ends, label_y):
        ax.plot([x * 1.08, label_x / 1.1], [y_end, y_label], color=color, linewidth=0.6, alpha=0.7)
        ax.text(label_x, y_label, text, va="center", fontsize=8, color=color)

    ax.set(xscale="log", xlim=(lives.min() / 1.5, label_x * 6), ylim=(0, top),
           xlabel="Cycles to failure $N_f$", ylabel=r"Stress amplitude $\sigma_a$ (MPa)")
    ax.xaxis.set_major_formatter(FuncFormatter(decade))
    ax.text(0.98, 0.97, f"loading mode: {mode_name}\nBasquin fit, drawn over tested stresses only",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=GREY)
    ax.grid(color="#eceae4", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "3_sn_curves")

    within_state["mode"] = mode_name
    within_state["curves"] = pd.DataFrame(rows).set_index("UTS (MPa)")
    return within_state, len(shown)


def main():
    # Load the same data as evaluate.py
    X, y, _, folds = load_clean(TRAIN)
    X_test, y_test, _, _ = load_clean(TEST)
    X_test = X_test[X.columns]          # same column order as training

    # Train table (with group and log10_Nf) needed for noise floor and figure 3
    train = pd.read_csv(TRAIN).rename(columns={"σa": "sigma_a"})

    print("Tuning (this is the slow part)...")
    tuned, summary = tune_all(X, y, folds)
    best_name = summary["rmse_mean"].idxmin()
    final_model = tuned[best_name]

    # Only the two models figure 1 shows
    preds = {"basquin_baseline": basquin_baseline(X, y, X_test),
             best_name: final_model.predict(X_test)}

    floor = repeat_test_scatter(train) # Noise floor from repeated tests
    # 30 repeats
    importance = feature_importance(final_model, X, y, folds, n_repeats=30)

    # Draw and save the three figures
    print("\nFigures")
    fig1_predicted_vs_measured(y_test, preds, best_name)
    kept = fig2_feature_importance(importance)
    within_state, drawn = fig3_sn_curves(train)

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

   
    print(f"\nFig 3, S-N curves (train, {drawn} of {within_state['n_states']} states drawn)")
    ladder = [
        ("Noise floor (repeat tests)", floor),
        (f"Basquin per state, in-sample ({within_state['n_states']} states)", within_state["per_state"]),
        (f"{MODEL_NAMES[best_name]}, CV on unseen states", cv_best),
        ("Basquin on sigma_a / UTS, in-sample", within_state["global_norm"]),
        ("Global Basquin on sigma_a, in-sample", within_state["global_raw"]),
    ]
    for label, value in sorted(ladder, key=lambda row: row[1]):
        print(f"  {label:48s} RMSE {value:.3f}  (miss factor {10 ** value:.1f})")
    print(f"\n  Curves drawn ({within_state['mode']}), b in sigma_a = sigma_f' (2Nf)^b")
    print(within_state["curves"].to_string())


if __name__ == "__main__":
    main()