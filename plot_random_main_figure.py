"""Main figure: random-direction detection vs angle from the concept subspace.

Two panels from the graded --sweep angle --random projection CSVs, presenting the
project's central result (detection of random unit directions depends on their angle
from the PCA concept subspace) the way it reads best:

  Panel A  k >= 2: thin per-k curves (sequential palette, light -> dark with k) plus a
           thick equal-weight mean curve over k. k=1 is excluded here on mathematical
           grounds, not cosmetic ones: a 1-D subspace collapses every random draw to
           +/-PC1, so its average mixes two categorically different injections.
  Panel B  k = 1: the same sweep split by that sign. Each draw's alpha=0 direction is
           sign(v.PC1)*PC1; the sign is reconstructed exactly from the stored
           random_seed and the concept's index over the subspace's test_names (the rng
           projection_experiment.py uses). The m-aligned pole (m = mean train-concept
           direction; m.PC1 ~ -0.8, so -PC1 here) is the concept-DENSE direction, the
           opposite pole the concept-SPARSE one. The poles converge at 90 degrees,
           where the injected direction has no PC1 component left and the sign stops
           existing.

Both panels use the coherent & detected metric (coherence AND affirmative_response)
and share the y axis. Three reference lines -- no injection, the same random vectors
at full strength, and the unfiltered all-concepts concept-vector rate -- are drawn on
both panels and direct-labeled (with their percentage) at the right edge of Panel A.
The selected "concepts identified >= 1/5 trials" baseline is deliberately left to the
appendix plots (plot_projection.py): it is a filtered subset, not a neutral reference.

Uncertainty: shaded bands are 95% cluster-bootstrap CIs resampling the base random
directions (one per concept x seed, the unit that is independent in this design; the
five trials and all angles of a draw are resampled together). The Panel A mean curve's
band re-computes the whole mean-over-k inside each resample -- the per-k curves are
deliberately drawn without their own bands to keep the panel readable.

    python plot_random_main_figure.py
    python plot_random_main_figure.py --csv new_results/..._anglesweep_random.csv \
        new_results/..._anglesweep_random_seed1.csv   # pool seeds once graded

Writes plots/main_figure_random_anglesweep.png and prints the per-panel tables.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from random_direction_analysis import draw_direction, load_unit_vectors, to_rate

# Sequential single-hue ramp (ColorBrewer Blues), light -> dark with k: order carries
# the k magnitude, the legend carries identity. Never cycled.
K_RAMP = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08306b"]
MEAN_COLOR = "#1a1a1a"
# Panel B is a polarity, not a magnitude: one warm / one cool (Okabe-Ito, CVD-safe).
# Labels are built at runtime: which PC1 pole is the concept-dense one is measured
# (sign of m.PC1, m = mean train-concept direction), never hardcoded.
POLE_COLORS = {"dense": "#D55E00", "sparse": "#0072B2"}
BASELINE_STYLES = {  # gray + dash pattern, recessive next to the data series
    "no injection": ("#999999", (0, (1, 1.8))),
    "random vectors": ("#333333", (0, (6, 1.5, 1, 1.5))),
    "concept vectors": ("#777777", (0, (2.5, 2.5))),
}


def coherent_detected(df):
    """0/1 coherent AND detected per row (NaN where either judge is ungraded)."""
    return to_rate(df["coherence_judge"]) * to_rate(df["affirmative_response_judge"])


def rate_matrix(df, ks, angles):
    """M[draw, k, angle] = that draw's mean coherent&detected rate (NaN when absent).
    A draw (concept x seed) is the independent unit: one base random direction."""
    draws = sorted(df["draw"].unique())
    index = {d: i for i, d in enumerate(draws)}
    M = np.full((len(draws), len(ks), len(angles)), np.nan)
    grouped = df.groupby(["draw", "k", "alpha"])["cd"].mean()
    for (d, k, a), v in grouped.items():
        M[index[d], ks.index(k), angles.index(a)] = v
    return M


def boot_ci(M, reduce, n_boot, rng):
    """95% percentile CI of reduce(M-resampled-over-axis-0). reduce keeps trailing axes."""
    stats = np.stack([reduce(M[rng.integers(0, len(M), len(M))]) for _ in range(n_boot)])
    return np.nanpercentile(stats, [2.5, 97.5], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Two-panel main figure for the random angle sweep")
    parser.add_argument("--csv", type=str, nargs="+",
                        default=["new_results/projection_results_pca_subspace_all_concepts"
                                 "_layer15_coeff6_anglesweep_random.csv"],
                        help="Graded --sweep angle --random CSVs (several seeds are pooled)")
    parser.add_argument("--no-injection-csv", type=str,
                        default="new_results/output_control_no_injection.csv")
    parser.add_argument("--all-concepts-csv", type=str,
                        default="new_results/output_all_concepts_layer15_coeff6.csv")
    parser.add_argument("--subspace", type=str,
                        default="pca_subspace_all_concepts_layer15_coeff6.npz")
    parser.add_argument("--vectors-dir", type=str, default="./vectors_local/llama",
                        help="Saved concept vectors; orient which PC1 pole faces concept content")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--out", type=str, default="plots/main_figure_random_anglesweep.png")
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    if not (df["sweep"].dropna() == "angle").all():
        raise SystemExit("Expected --sweep angle CSVs (alpha column = sweep fraction t).")
    df = df[df["alpha"].notna()].copy()
    df["cd"] = coherent_detected(df)
    df["draw"] = list(zip(df["random_seed"].astype(int), df["concept"]))
    ks = sorted(int(k) for k in df["k"].unique() if k >= 2)
    angles = sorted(df["alpha"].unique())          # sweep fraction t; nominal angle = t*90
    degs = [a * 90 for a in angles]
    rng = np.random.default_rng(0)

    # ---- geometry: the sign of each draw's PC1 component, oriented by the train mean ----
    d = np.load(args.subspace, allow_pickle=True)
    comps = d["components"].astype(np.float32)
    test_names = [str(w) for w in d["test_names"]]
    concept_index = {c: i for i, c in enumerate(test_names)}
    pc1 = comps[0] / np.linalg.norm(comps[0])
    unit = load_unit_vectors(args.vectors_dir, int(d["layer"]), str(d["vec_type"]))
    m = np.sum([unit[c] for c in {str(w) for w in d["train_names"]} if c in unit], axis=0)
    m_dot_pc1 = float((m / np.linalg.norm(m)) @ pc1)
    dense_sign = 1.0 if m_dot_pc1 > 0 else -1.0
    print(f"mean train direction . PC1 = {m_dot_pc1:+.2f} -> concept-dense pole is "
          f"{'+' if dense_sign > 0 else '-'}PC1")
    pole = {dr: ("dense" if np.sign(float(
                draw_direction(dr[0], concept_index[dr[1]], comps.shape[1]) @ pc1))
                == dense_sign else "sparse")
            for dr in df["draw"].unique()}
    pole_labels = {
        "dense": f"concept-dense direction ({'+' if dense_sign > 0 else '-'}PC1 pole)",
        "sparse": f"concept-sparse direction ({'-' if dense_sign > 0 else '+'}PC1 pole)",
    }

    # ---- baselines (coherent & detected, like everything else) ----
    baselines = {
        "no injection": coherent_detected(pd.read_csv(args.no_injection_csv)),
        "random vectors": coherent_detected(
            pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
            .query("variant == 'full'")),
        "concept vectors": coherent_detected(pd.read_csv(args.all_concepts_csv)),
    }
    for name, r in baselines.items():
        r = r.dropna()
        print(f"baseline {name}: {r.mean():.3f} (n={len(r)})")
        baselines[name] = float(r.mean())

    # ---- figure ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)

    # Panel A: k >= 2 thin curves + bootstrap-banded mean over k
    M = rate_matrix(df[df["k"] >= 2], ks, angles)
    per_k = np.nanmean(M, axis=0)                       # [k, angle]
    mean_k = per_k.mean(axis=0)
    lo, hi = boot_ci(M, lambda S: np.nanmean(S, axis=0).mean(axis=0), args.n_boot, rng)
    for i, k in enumerate(ks):
        ax_a.plot(degs, per_k[i], color=K_RAMP[i], linewidth=1.4, alpha=0.85,
                  marker="o", markersize=3.5, label=f"k = {k}", zorder=3)
    ax_a.fill_between(degs, lo, hi, color=MEAN_COLOR, alpha=0.15, linewidth=0, zorder=2)
    ax_a.plot(degs, mean_k, color=MEAN_COLOR, linewidth=2.8, marker="o", markersize=5,
              label="mean over k", zorder=4)
    ax_a.set_title("Higher-dimensional subspaces (k ≥ 2)", fontsize=12)
    ax_a.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Panel B: k = 1 split by the sign of the draw's PC1 component
    k1 = df[df["k"] == 1].copy()
    k1["pole"] = k1["draw"].map(pole)
    for p in ("dense", "sparse"):
        Mp = rate_matrix(k1[k1["pole"] == p], [1], angles)
        est = np.nanmean(Mp, axis=0)[0]
        lo_p, hi_p = boot_ci(Mp, lambda S: np.nanmean(S, axis=0)[0], args.n_boot, rng)
        n_draws = (k1["pole"] == p).groupby(k1["draw"]).any().sum()
        ax_b.fill_between(degs, lo_p, hi_p, color=POLE_COLORS[p], alpha=0.15,
                          linewidth=0, zorder=2)
        ax_b.plot(degs, est, color=POLE_COLORS[p], linewidth=2.4, marker="o",
                  markersize=5, label=f"{pole_labels[p]}, {n_draws} draws", zorder=3)
        print(f"\nk=1 {p} pole rates by angle:")
        print("  " + "  ".join(f"{deg:.0f}deg {v:.2f}" for deg, v in zip(degs, est)))
    ax_b.set_title("PC1 (k = 1) is orientation-dependent", fontsize=12)
    # Lower right is the one corner both poles and their bands leave free.
    ax_b.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # Shared dressing: baselines on both panels; the percent labels sit at the right
    # edge of Panel B, where the curves have converged and left the margin free (in
    # Panel A the data crosses every baseline near the right edge).
    for ax in (ax_a, ax_b):
        for name, y in baselines.items():
            color, dashes = BASELINE_STYLES[name]
            ax.axhline(y, color=color, linestyle=dashes, linewidth=1.4, zorder=1)
        ax.set_xlim(-3, 93)
        ax.set_xticks(range(0, 91, 15))
        ax.set_xticklabels(["0°\nin subspace", "15°", "30°", "45°",
                            "60°", "75°", "90°\northogonal"], fontsize=9)
        ax.set_xlabel("Angle to the concept subspace", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    for name, y in baselines.items():
        ax_b.text(91, y + 0.008, f"{name}: {y:.0%}", ha="right", va="bottom",
                  fontsize=8, color=BASELINE_STYLES[name][0], zorder=4)
    # Truncated y axis (plainly visible limits) so the ~10-15 pp effect fills the frame
    ax_a.set_ylim(0.05, 0.72)
    ax_a.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_a.set_yticks(np.arange(0.1, 0.71, 0.1))
    ax_a.set_ylabel("Coherent & detected rate", fontsize=11)

    # "more detectable near the subspace" rather than "decreases as they leave": an
    # unconstrained random vector is already near-orthogonal (~sqrt(k/4096) of its norm
    # in-subspace), so proximity is the special condition, not departure.
    fig.suptitle("Random unit directions are more detectable near the concept subspace",
                 fontsize=13.5)
    fig.text(0.01, 0.005,
             f"Bands: pointwise 95% cluster-bootstrap CIs ({args.n_boot} resamples of the "
             "base random directions; one per concept × seed, all angles and trials of a "
             "draw resampled together). Coverage is per angle, not simultaneous.",
             fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out}")

    print("\nPanel A rates (rows: k, cols: nominal angle):")
    print(pd.DataFrame(per_k, index=ks, columns=[f"{x:.0f}" for x in degs]).round(2).to_string())
    print("mean over k: " + "  ".join(f"{v:.2f}" for v in mean_k))


if __name__ == "__main__":
    main()
