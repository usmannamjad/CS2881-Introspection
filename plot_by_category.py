"""Plot detection/coherence rates broken down by concept category.

The all_concepts run injects vectors for every word list in dataset/simple_data.json
(the 50 main concept_vector_words + 5 categories x 50). The results CSV stores only the
concept name, so this script maps each concept back to its source word list and plots
the per-category rate for each judge with binomial standard-error bars.

    python plot_by_category.py --csv new_results/output_all_concepts_layer15_coeff6.csv

Judges default to coherence + affirmative_response_followed_by_correct_identification
(the two the all_concepts run grades). Writes plots/main_figure_<run>_by_category.png
plus a companion two-panel figure (<...>_hitcount.png): overall coherent & identified
rate per category with 95% cluster-bootstrap CIs (concepts resampled, trials kept
together) above the stacked per-concept hit-count distribution. Prints per-category
summary tables.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# Embed TrueType (Type 42) fonts so camera-ready checkers don't flag Type 3
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Display order for the x-axis (baseline_words is not injected, so it's excluded).
# The five homogeneous categories come first; the mixed concept_vector_words list last.
CATEGORY_KEYS = [
    "famous_people", "countries",
    "concrete_nouns", "abstract_nouns", "verbs",
    "concept_vector_words",
]
CATEGORY_LABELS = {
    "concept_vector_words": "mixed concepts",
    "famous_people": "famous people",
    "countries": "countries",
    "concrete_nouns": "concrete nouns",
    "abstract_nouns": "abstract nouns",
    "verbs": "verbs",
}

# Two Okabe-Ito colors: colorblind-safe and well separated. Identity is also carried by
# the legend, never color alone.
JUDGE_STYLE = {
    "coherence": ("#0072B2", "coherence"),
    "affirmative_response_followed_by_correct_identification": ("#E69F00", "detected + identified"),
    "affirmative_response": ("#009E73", "detected (affirmative)"),
    "thinking_about_word": ("#CC79A7", "thinking about word"),
}


def to_rate(series):
    """Judge column values as 0/1 floats (NaN for ungraded), robust to CSV round-trips."""
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1.0: 1.0, 0.0: 0.0})


def cluster_bootstrap_ci(per_concept_means, n_boot=10_000, seed=0):
    """95% CI for a category mean, clustering on concept.

    Resamples concepts with replacement (each draw keeps all of that concept's
    trials together via its per-concept mean), recomputes the category mean, and
    returns the 2.5th/97.5th percentiles over n_boot replicates.
    """
    arr = np.asarray(per_concept_means, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def build_concept_category(dataset_path):
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    concept_to_category = {}
    for key in CATEGORY_KEYS:
        for word in data[key]:
            concept_to_category[word] = key
    return concept_to_category


def main():
    parser = argparse.ArgumentParser(description="Plot judge rates grouped by concept category")
    parser.add_argument("--csv", type=str, default="new_results/output_all_concepts_layer15_coeff6.csv",
                        help="Results CSV from the all_concepts run")
    parser.add_argument("--dataset", type=str, default="dataset/simple_data.json",
                        help="simple_data.json used to map concept -> category")
    parser.add_argument("--judges", type=str, nargs="+",
                        default=["coherence", "affirmative_response_followed_by_correct_identification"],
                        help="Judge columns to plot (default: coherence + detected/identified)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG (default: plots/main_figure_<run>_by_category.png)")
    parser.add_argument("--success-judge", type=str,
                        default="affirmative_response_followed_by_correct_identification",
                        help="Judge counted as a 'correct identification' in the per-concept "
                             "hit-count distribution (default: detected + identified)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    run_label = csv_path.stem.removeprefix("output_")
    df = pd.read_csv(csv_path)

    concept_to_category = build_concept_category(args.dataset)
    df["category"] = df["concept"].map(concept_to_category)
    unmapped = df[df["category"].isna()]["concept"].unique()
    if len(unmapped):
        print(f"Warning: {len(unmapped)} concept(s) not found in any category, dropped: {sorted(unmapped)[:10]}")
    df = df[df["category"].notna()]

    # Only keep categories that actually appear, in the fixed order
    categories = [c for c in CATEGORY_KEYS if c in set(df["category"])]

    # Per (category, judge): mean rate, sample count, binomial standard error
    stats = {}  # (judge, category) -> (rate, n, se)
    for judge in args.judges:
        col = f"{judge}_judge"
        if col not in df:
            print(f"Skipping {judge}: column {col} not in CSV")
            continue
        rates = to_rate(df[col])
        grouped = rates.groupby(df["category"]).agg(["mean", "count"])
        for category in categories:
            if category in grouped.index and grouped.loc[category, "count"] > 0:
                mean = grouped.loc[category, "mean"]
                n = int(grouped.loc[category, "count"])
                se = (mean * (1 - mean) / n) ** 0.5
                stats[(judge, category)] = (mean, n, se)

    plotted_judges = [j for j in args.judges if any((j, c) in stats for c in categories)]
    if not plotted_judges:
        raise SystemExit("No judge data to plot in this CSV.")

    # Grouped bars: one group per category, one bar per judge
    x = range(len(categories))
    n_judges = len(plotted_judges)
    bar_width = 0.8 / n_judges

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(categories)), 6))
    for j_idx, judge in enumerate(plotted_judges):
        color, label = JUDGE_STYLE.get(judge, ("#666666", judge))
        heights = [stats.get((judge, c), (0.0, 0, 0.0))[0] for c in categories]
        errs = [stats.get((judge, c), (0.0, 0, 0.0))[2] for c in categories]
        # Offset each judge's bars within the group; 2px-equivalent gap via bar_width < slot
        offsets = [xi + (j_idx - (n_judges - 1) / 2) * bar_width for xi in x]
        bars = ax.bar(offsets, heights, width=bar_width * 0.92, yerr=errs, capsize=3,
                      color=color, label=label, edgecolor="white", linewidth=0.5)
        # Direct-label each bar with its rate, placed above the error-bar cap so the
        # text never overlaps the whisker (selective: only the value, kept small)
        for rect, h, e in zip(bars, heights, errs):
            ax.text(rect.get_x() + rect.get_width() / 2, h + e + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#333333")

    # Per-category sample size (concepts x trials); same across judges that ran
    n_per_cat = {c: max((stats[(j, c)][1] for j in plotted_judges if (j, c) in stats), default=0)
                 for c in categories}
    xticklabels = [f"{CATEGORY_LABELS.get(c, c)}\n(n={n_per_cat[c]})" for c in categories]

    layer = df["layer"].iloc[0]
    coeff = df["coeff"].iloc[0]
    temperature = df["temperature"].iloc[0] if "temperature" in df else "?"
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_ylabel("Rate (mean over concepts x trials)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{run_label} by category (layer={layer}, coeff={coeff}, temp={temperature})", fontsize=13)
    # Coherence bars fill the top and identification bars sit low, leaving the mid-band
    # empty; park the legend there so it never overlaps a bar or its value label.
    ax.legend(fontsize=10, loc="center", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else plots_dir / f"main_figure_{run_label}_by_category.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # Per-category summary table
    print("\nPer-category rates:")
    header = "category".ljust(18) + "".join(JUDGE_STYLE.get(j, (None, j))[1][:22].ljust(24) for j in plotted_judges)
    print(header)
    for c in categories:
        row = CATEGORY_LABELS.get(c, c).ljust(18)
        for j in plotted_judges:
            if (j, c) in stats:
                rate, n, se = stats[(j, c)]
                row += f"{rate:.3f} +/- {se:.3f} (n={n})".ljust(24)
            else:
                row += "-".ljust(24)
        print(row)

    # Per-concept hit-count distribution: for each concept, how many of its trials were
    # coherent AND marked correct by the success judge, then how many concepts land on
    # each hit count (0..max_trials), broken down by category.
    success_col = f"{args.success_judge}_judge"
    if success_col in df:
        # NaN in either judge propagates, so ungraded rows stay excluded from rates
        success = to_rate(df["coherence_judge"]) * to_rate(df[success_col])
        hits = success.fillna(0.0).groupby([df["concept"], df["category"]]).sum()
        hits = hits.reset_index(name="hits")
        hits["hits"] = hits["hits"].astype(int)
        max_trials = int(df.groupby("concept")["trial"].nunique().max())
        counts = list(range(max_trials + 1))

        # dist[(category, k)] = number of concepts in category with exactly k correct IDs
        dist = {(c, k): 0 for c in categories for k in counts}
        for _, r in hits.iterrows():
            dist[(r["category"], r["hits"])] += 1

        success_label = "coherent & " + JUDGE_STYLE.get(args.success_judge, (None, args.success_judge))[1]
        print(f"\nConcepts by number of correct identifications out of {max_trials} "
              f"trials ({success_label}):")
        header = "category".ljust(18) + "".join(f"{k}x".rjust(6) for k in counts) + "  total".rjust(8)
        print(header)
        for c in categories:
            row_counts = [dist[(c, k)] for k in counts]
            row = CATEGORY_LABELS.get(c, c).ljust(18) + "".join(str(v).rjust(6) for v in row_counts)
            row += str(sum(row_counts)).rjust(8)
            print(row)
        total_counts = [sum(dist[(c, k)] for c in categories) for k in counts]
        print("all".ljust(18) + "".join(str(v).rjust(6) for v in total_counts)
              + str(sum(total_counts)).rjust(8))

        # Per-category overall rate (equal weight per concept, matching the cluster
        # bootstrap) with 95% cluster-bootstrap CIs; concepts are the resampling unit.
        concept_rates = success.groupby([df["concept"], df["category"]]).mean()
        concept_rates = concept_rates.reset_index(name="rate")
        cat_rate, cat_ci = {}, {}
        for c in categories:
            vals = concept_rates.loc[concept_rates["category"] == c, "rate"].dropna().to_numpy()
            cat_rate[c] = float(vals.mean())
            cat_ci[c] = cluster_bootstrap_ci(vals)

        print(f"\nOverall {success_label} rate per category (95% cluster-bootstrap CI):")
        for c in categories:
            lo, hi = cat_ci[c]
            print(f"  {CATEGORY_LABELS.get(c, c).ljust(18)}{cat_rate[c]:.3f} "
                  f"[{lo:.3f}, {hi:.3f}] (n={len(hits[hits['category'] == c])} concepts)")

        # Two companion figures: overall rate with bootstrap CIs, and the stacked
        # hit-count distribution, saved separately for the writeup/slides.
        cmap = plt.get_cmap("viridis")
        seg_colors = [cmap(i / max(1, max_trials)) for i in counts]
        cat_x = range(len(categories))
        xticklabels_plain = [CATEGORY_LABELS.get(c, c) for c in categories]

        # ICML single-column figure: rendered at final physical size (~3.25 in wide)
        # with fonts chosen for that size, so LaTeX includes it without rescaling.
        # No in-figure title -- the description belongs in the \caption{}.
        fig_rate, ax_rate = plt.subplots(figsize=(3.25, 2.5))
        rates = [cat_rate[c] for c in categories]
        yerr = [[cat_rate[c] - cat_ci[c][0] for c in categories],
                [cat_ci[c][1] - cat_rate[c] for c in categories]]
        ax_rate.bar(cat_x, rates, width=0.7, color="#E69F00",
                           edgecolor="white", linewidth=0.5,
                           yerr=yerr, capsize=2, ecolor="#333333",
                           error_kw={"linewidth": 1.0})
        ax_rate.set_title("Correct identification varies strongly\nacross concept categories",
                          fontsize=8)
        ax_rate.set_ylabel("Correct identification rate", fontsize=8)
        ax_rate.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax_rate.set_ylim(0, max(cat_ci[c][1] for c in categories) * 1.25)
        ax_rate.set_xticks(list(cat_x))
        ax_rate.set_xticklabels(xticklabels_plain, fontsize=7, rotation=30, ha="right")
        ax_rate.tick_params(axis="y", labelsize=7)
        ax_rate.grid(True, axis="y", alpha=0.3)
        ax_rate.set_axisbelow(True)
        fig_rate.tight_layout()
        rate_path = out_path.with_name(out_path.stem + "_rate.pdf")
        fig_rate.savefig(rate_path, bbox_inches="tight")
        fig_rate.savefig(rate_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig_rate)
        print(f"\nRate figure saved to {rate_path} (+ .png preview)")

        fig2, ax2 = plt.subplots(figsize=(max(10, 1.6 * len(categories)), 6))
        bottoms = [0] * len(categories)
        for k in counts:
            heights = [dist[(c, k)] for c in categories]
            ax2.bar(cat_x, heights, bottom=bottoms, width=0.7, color=seg_colors[k],
                    edgecolor="white", linewidth=0.5, label=f"{k} correct")
            for xi, (h, b) in enumerate(zip(heights, bottoms)):
                if h > 0:
                    ax2.text(xi, b + h / 2, str(h), ha="center", va="center",
                             fontsize=8, color="white")
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax2.set_xticks(list(cat_x))
        ax2.set_xticklabels(xticklabels_plain, fontsize=10)
        ax2.set_ylabel("Number of concepts", fontsize=12)
        # Legend outside the axes so it never covers the full-height stacks
        ax2.legend(title="correct IDs", fontsize=9, loc="upper left",
                   bbox_to_anchor=(1.01, 1.0), framealpha=0.9)
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.set_axisbelow(True)

        ax2.set_title("Distribution of correct identifications by concept category",
                      fontsize=14)
        fig2.tight_layout()
        dist_path = out_path.with_name(out_path.stem + "_hitcount.png")
        fig2.savefig(dist_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"\nHit-count figure saved to {dist_path}")
    else:
        print(f"\nSkipping hit-count distribution: column {success_col} not in CSV")


if __name__ == "__main__":
    main()
