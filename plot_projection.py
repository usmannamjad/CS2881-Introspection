"""Plot the projection sweep: identification/detection rate vs alpha, one line per k.

Reads a GRADED projection_results_*.csv (from projection_experiment.py + judge_results.py)
and draws, per judge, how the judge's rate changes as the injected direction rotates from
inside the PCA subspace (alpha=0, proj) to orthogonal to it (alpha=1, residual). One line
per subspace dim k, with the raw `full` vector drawn as a horizontal baseline.

Read it as: a curve that starts high near alpha=0 and falls toward alpha=1 means the
judged behaviour lives in the subspace. Detection (affirmative_response) is a weaker signal
than identification, so its curve should sit higher / fall less -- the two subplots make
that comparison directly.

    python plot_projection.py --csv projection_results_pca_subspace_all_concepts_layer15_coeff6.csv

Writes plots/projection_curves_<run>.png and prints the per-judge, per-k table.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Weakest signal first: detection (noticed a thought) then identification (named it).
DETECT_JUDGE = "affirmative_response"
SUCCESS_JUDGE = "affirmative_response_followed_by_correct_identification"
DEFAULT_JUDGES = [DETECT_JUDGE, SUCCESS_JUDGE]

JUDGE_LABELS = {
    "affirmative_response": "detected (affirmative)",
    "affirmative_response_followed_by_correct_identification": "detected + identified",
    "coherence": "coherence",
    "thinking_about_word": "thinking about word",
}

# Okabe-Ito, colorblind-safe, assigned to k in fixed order (never cycled). Identity is
# always carried by the legend too, never color alone.
K_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
BASELINE_COLOR = "#555555"


def to_rate(series):
    """Judge column values as 0/1 floats (NaN for ungraded), robust to CSV round-trips."""
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1.0: 1.0, 0.0: 0.0})


def main():
    parser = argparse.ArgumentParser(description="Plot projection sweep rate vs alpha per k")
    parser.add_argument("--csv", type=str, required=True,
                        help="Graded projection_results_*.csv")
    parser.add_argument("--judges", type=str, nargs="+", default=DEFAULT_JUDGES,
                        help="Judge columns to plot, one subplot each (default: detection + identification)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG (default: plots/projection_curves_<run>.png)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    run_label = csv_path.stem.removeprefix("projection_results_").removeprefix("output_")
    df = pd.read_csv(csv_path)

    ks = sorted(int(k) for k in df.loc[df["k"].notna(), "k"].unique())
    alphas = sorted(a for a in df.loc[df["variant"] != "full", "alpha"].dropna().unique())

    # Keep only judges that are actually graded in this CSV.
    judges = []
    for j in args.judges:
        col = f"{j}_judge"
        if col in df and to_rate(df[col]).notna().any():
            judges.append(j)
        else:
            print(f"Skipping {j}: column {col} missing or ungraded.")
    if not judges:
        raise SystemExit("No graded judge columns to plot. Grade the CSV with judge_results.py first.")

    fig, axes = plt.subplots(1, len(judges), figsize=(6.2 * len(judges), 5.4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, judge in zip(axes, judges):
        rate = to_rate(df[f"{judge}_judge"])
        # full baseline (alpha is NaN for these rows): a horizontal reference line.
        full_mask = df["variant"] == "full"
        full_rate = rate[full_mask].mean()
        full_n = int(rate[full_mask].notna().sum())
        ax.axhline(full_rate, color=BASELINE_COLOR, linestyle=(0, (4, 3)), linewidth=1.6,
                   label=f"full vector ({full_rate:.2f})", zorder=2)

        for i, k in enumerate(ks):
            color = K_COLORS[i % len(K_COLORS)]
            means, errs, xs = [], [], []
            for a in alphas:
                sel = (df["k"] == k) & (df["alpha"] == a)
                r = rate[sel].dropna()
                if r.empty:
                    continue
                p, n = r.mean(), len(r)
                xs.append(a)
                means.append(p)
                errs.append((p * (1 - p) / n) ** 0.5)   # binomial standard error
            if xs:
                ax.errorbar(xs, means, yerr=errs, marker="o", markersize=7, linewidth=2,
                            capsize=3, color=color, label=f"k={k}", zorder=3)

        ax.set_title(JUDGE_LABELS.get(judge, judge), fontsize=12)
        ax.set_xlabel("alpha  (0 = in subspace  ->  1 = orthogonal)", fontsize=10)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        # Anchor the ends of the sweep so proj/residual are unmistakable.
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.2f}" for a in alphas], fontsize=8)
        ax.annotate("proj", (0, 0), xytext=(0, -34), textcoords="offset points",
                    ha="center", fontsize=8, color=BASELINE_COLOR)
        ax.annotate("residual", (1, 0), xytext=(0, -34), textcoords="offset points",
                    ha="center", fontsize=8, color=BASELINE_COLOR)
        ax.legend(fontsize=9, loc="best", framealpha=0.9, title="subspace dim")

    axes[0].set_ylabel("Rate (mean over concepts x trials)", fontsize=11)
    fig.suptitle(f"Projection sweep: {run_label}  (n_full={full_n} trials)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else plots_dir / f"projection_curves_{run_label}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # Per-judge, per-k table (rows: k, cols: alpha) so the numbers back the picture.
    for judge in judges:
        rate = to_rate(df[f"{judge}_judge"])
        print(f"\n{JUDGE_LABELS.get(judge, judge)} rate (rows: k, cols: alpha):")
        pivot = (rate.groupby([df["k"], df["alpha"]]).mean().unstack().round(2))
        print(pivot.to_string())


if __name__ == "__main__":
    main()
