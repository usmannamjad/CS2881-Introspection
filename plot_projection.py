"""Plot the projection sweep: judge rate vs alpha (and vs actual angle), one line per k.

Reads a GRADED projection_results_*.csv (from projection_experiment.py + judge_results.py)
and draws, per judge, how the judge's rate changes as the injected direction rotates from
inside the PCA subspace (alpha=0, proj) to orthogonal to it (alpha=1, residual). One line
per subspace dim k, with the raw `full` vector drawn as a horizontal baseline.

Judges can be combined with '+' (logical AND of the graded booleans): e.g.
`coherence+affirmative_response` counts trials that are BOTH coherent AND detected. The
default panels are detection and identification alone, then coherence, then coherence
ANDed with each -- the AND panels separate "introspected" from "rambled incoherently and
the judge pattern-matched".

alpha is NOT linear in angle: proj and residual have different norms, so equal alpha steps
sweep unequal angles. Injection re-normalizes every variant, but normalization only
rescales a vector -- it never rotates it -- so the angle between the injected direction
and the subspace is well defined: theta = atan2(alpha*||residual||, (1-alpha)*||proj||),
running 0 deg (inside the subspace) -> 90 deg (orthogonal). A second figure is written
with that angle on the x-axis: each curve point sits at the mean over concepts of that
concept's exact angle (||proj|| differs per concept, so the same alpha lands at different
angles). Newer CSVs carry the angle in an
'angle' column (written by projection_experiment.py; the only correct source for --random
runs); for older CSVs it is recomputed from the subspace .npz + vector files when those
are available (defaults match pca.py / projection_experiment.py).

    python plot_projection.py --csv projection_results_pca_subspace_all_concepts_layer15_coeff6.csv

Writes plots/projection_curves_<run>.png (+ projection_curves_<run>_angle.png) and prints
the per-judge, per-k tables.
"""
import argparse
from math import atan2, ceil, degrees
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Weakest signal first: detection (noticed a thought) then identification (named it).
DETECT_JUDGE = "affirmative_response"
SUCCESS_JUDGE = "affirmative_response_followed_by_correct_identification"
COHERENCE_JUDGE = "coherence"
DEFAULT_JUDGES = [
    DETECT_JUDGE,
    SUCCESS_JUDGE,
    COHERENCE_JUDGE,
    f"{COHERENCE_JUDGE}+{DETECT_JUDGE}",
    f"{COHERENCE_JUDGE}+{SUCCESS_JUDGE}",
]

JUDGE_LABELS = {
    "affirmative_response": "detected (affirmative)",
    "affirmative_response_followed_by_correct_identification": "detected + identified",
    "coherence": "coherent",
    "thinking_about_word": "thinking about word",
    f"{COHERENCE_JUDGE}+{DETECT_JUDGE}": "coherent & detected",
    f"{COHERENCE_JUDGE}+{SUCCESS_JUDGE}": "coherent & detected + identified",
}

# Okabe-Ito, colorblind-safe, assigned to k in fixed order (never cycled). Identity is
# always carried by the legend too, never color alone.
K_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
BASELINE_COLOR = "#555555"
MAX_COLS = 3


def to_rate(series):
    """Judge column values as 0/1 floats (NaN for ungraded), robust to CSV round-trips."""
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1.0: 1.0, 0.0: 0.0})


def spec_rate(df, spec):
    """0/1 rate for a judge spec; 'a+b' = AND of the component judges (NaN if any ungraded)."""
    rate = None
    for part in spec.split("+"):
        r = to_rate(df[f"{part}_judge"])
        rate = r if rate is None else rate * r
    return rate


def spec_label(spec):
    if spec in JUDGE_LABELS:
        return JUDGE_LABELS[spec]
    return " & ".join(JUDGE_LABELS.get(p, p) for p in spec.split("+"))


def subspace_angles(df, subspace_path, vectors_dir, ks, alphas):
    """Angle in degrees between each concept's injected direction and the subspace.

    Rebuilds proj/residual exactly as projection_experiment.build_variants does; the
    injection's re-normalization only rescales, so angles are unaffected. Returns a
    DataFrame [concept, k, alpha, angle] (one row per concept, not aggregated), or None
    if the subspace .npz or all vector files are missing.
    """
    if not Path(subspace_path).exists():
        return None
    import torch

    d = np.load(subspace_path, allow_pickle=True)
    components = d["components"].astype(np.float32)
    mean = d["mean"].astype(np.float32)
    layer, vec_type = int(d["layer"]), str(d["vec_type"])

    rows = []
    for concept in df["concept"].dropna().unique():
        fp = Path(vectors_dir) / f"{concept}_{layer}_{vec_type}.pt"
        if not fp.exists():
            continue
        raw = torch.load(fp, weights_only=False)["vector"]
        v = np.asarray(raw.detach().cpu().float() if isinstance(raw, torch.Tensor) else raw,
                       dtype=np.float32).ravel()
        n = np.linalg.norm(v)
        if n == 0:
            continue
        v = v / n
        for k in ks:
            P = components[:k]
            proj = mean + (v - mean) @ P.T @ P
            residual = v - proj
            for a in alphas:
                w = (1.0 - a) * proj + a * residual
                w_in = (w @ P.T) @ P            # component inside span(P)
                rows.append((concept, k, a,
                             degrees(atan2(np.linalg.norm(w - w_in), np.linalg.norm(w_in)))))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["concept", "k", "alpha", "angle"])


def draw_figure(df, judges, ks, alphas, angle_df, out_path, run_label):
    """One panel per judge spec, one line per k. angle_df=None -> x is alpha; else x is the
    angle to the subspace in degrees, with each concept's rate scattered at its own angle."""
    ncols = min(MAX_COLS, len(judges))
    nrows = ceil(len(judges) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.4 * nrows),
                             sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(judges):]:
        ax.set_visible(False)

    full_mask = df["variant"] == "full"
    full_n = int(spec_rate(df, judges[0])[full_mask].notna().sum())

    for ax, judge in zip(flat, judges):
        rate = spec_rate(df, judge)
        # full baseline (alpha is NaN for these rows): a horizontal reference line.
        full_rate = rate[full_mask].mean()
        ax.axhline(full_rate, color=BASELINE_COLOR, linestyle=(0, (4, 3)), linewidth=1.6,
                   label=f"full vector ({full_rate:.2f})", zorder=2)

        for i, k in enumerate(ks):
            color = K_COLORS[i % len(K_COLORS)]
            angle_of = None
            if angle_df is not None:
                ka = angle_df[angle_df["k"] == k]
                angle_of = {(c, a): deg
                            for c, a, deg in zip(ka["concept"], ka["alpha"], ka["angle"])}
            xs, means, errs = [], [], []
            for a in alphas:
                sel = (df["k"] == k) & (df["alpha"] == a)
                r = rate[sel].dropna()
                if r.empty:
                    continue
                if angle_of is None:
                    x = a
                else:
                    # Curve point sits at the mean over concepts of each concept's exact
                    # angle (||proj|| differs per concept, so the same alpha lands at
                    # different angles).
                    angles = [angle_of[(c, a)] for c in df.loc[r.index, "concept"].unique()
                              if (c, a) in angle_of]
                    if not angles:
                        continue
                    x = float(np.mean(angles))
                p, n = r.mean(), len(r)
                xs.append(x)
                means.append(p)
                errs.append((p * (1 - p) / n) ** 0.5)   # binomial standard error
            if xs:
                ax.errorbar(xs, means, yerr=errs, marker="o", markersize=7, linewidth=2,
                            capsize=3, color=color, label=f"k={k}", zorder=3)

        ax.set_title(spec_label(judge), fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        # Anchor the ends of the sweep so proj/residual are unmistakable.
        if angle_df is None:
            ax.set_xlabel("alpha  (0 = in subspace  ->  1 = orthogonal)", fontsize=10)
            ax.set_xlim(-0.04, 1.04)
            ax.set_xticks(alphas)
            ax.set_xticklabels([f"{a:.2f}" for a in alphas], fontsize=8)
            ends = [("proj", 0), ("residual", 1)]
        else:
            ax.set_xlabel("angle to subspace (degrees)", fontsize=10)
            ax.set_xlim(-3, 93)
            ax.set_xticks(range(0, 91, 15))
            ends = [("proj", 0), ("residual", 90)]
        for text, x in ends:
            ax.annotate(text, (x, 0), xytext=(0, -34), textcoords="offset points",
                        ha="center", fontsize=8, color=BASELINE_COLOR)
        ax.legend(fontsize=9, loc="best", framealpha=0.9, title="subspace dim")

    for row in axes:
        row[0].set_ylabel("Rate (mean over concepts x trials)", fontsize=11)
    fig.suptitle(f"Projection sweep: {run_label}  (n_full={full_n} trials)", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot projection sweep rate vs alpha/angle per k")
    parser.add_argument("--csv", type=str, required=True,
                        help="Graded projection_results_*.csv")
    parser.add_argument("--judges", type=str, nargs="+", default=DEFAULT_JUDGES,
                        help="Judge columns to plot, one subplot each; combine with '+' for "
                             "logical AND, e.g. coherence+affirmative_response "
                             "(default: detection, identification, coherence, and both ANDs)")
    parser.add_argument("--subspace", type=str, default=None,
                        help="pca_subspace_*.npz for the angle-axis figure (default: derived "
                             "from the CSV name; angle figure is skipped if not found)")
    parser.add_argument("--vectors-dir", type=str, default="./vectors_local/llama",
                        help="Directory of saved {concept}_{layer}_{vec_type}.pt files, "
                             "needed for the angle-axis figure")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG (default: plots/projection_curves_<run>.png; the "
                             "angle figure gets an _angle suffix)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    run_label = csv_path.stem.removeprefix("projection_results_").removeprefix("output_")
    df = pd.read_csv(csv_path)

    ks = sorted(int(k) for k in df.loc[df["k"].notna(), "k"].unique())
    alphas = sorted(a for a in df.loc[df["variant"] != "full", "alpha"].dropna().unique())

    # Keep only judge specs whose every component is actually graded in this CSV.
    judges = []
    for spec in args.judges:
        parts = spec.split("+")
        if all(f"{p}_judge" in df and to_rate(df[f"{p}_judge"]).notna().any() for p in parts):
            judges.append(spec)
        else:
            print(f"Skipping {spec}: a component column is missing or ungraded.")
    if not judges:
        raise SystemExit("No graded judge columns to plot. Grade the CSV with judge_results.py first.")

    plots_dir = Path("plots")
    out_path = Path(args.out) if args.out else plots_dir / f"projection_curves_{run_label}.png"
    draw_figure(df, judges, ks, alphas, None, out_path, run_label)

    # Angle-axis version: same curves, x = actual degrees between direction and subspace.
    # Newer CSVs store the exact per-row angle, so no subspace/.pt files are needed -- and
    # it is the only correct source for --random runs, whose injected directions aren't on
    # disk (the .pt files hold the CONCEPT vectors, so recomputing would give wrong angles).
    if "angle" in df.columns and df["angle"].notna().any():
        angle_df = (df.loc[df["variant"] != "full", ["concept", "k", "alpha", "angle"]]
                    .dropna(subset=["angle"]).drop_duplicates())
    elif "vector_source" in df.columns and (df["vector_source"] == "random").any():
        print("Random-direction run without a stored angle column: skipping the angle-axis "
              "figure (vector files hold the concept vectors, not the injected random "
              "directions).")
        angle_df = None
    else:
        if args.subspace:
            subspace_path = Path(args.subspace)
        else:
            name = run_label if run_label.startswith("pca_subspace") else f"pca_subspace_{run_label}"
            subspace_path = Path(f"{name}.npz")
            if not subspace_path.exists():
                subspace_path = csv_path.parent / subspace_path.name
        angle_df = subspace_angles(df, subspace_path, args.vectors_dir, ks, alphas)
        if angle_df is None:
            print(f"No subspace/vectors found ({subspace_path}); skipping the angle-axis figure. "
                  f"This CSV predates the stored 'angle' column, so the subspace .npz and the "
                  f".pt vector files are needed to recompute angles. If the fit ran on Modal, "
                  f"download them first:\n"
                  f"    modal volume get introspection-results {subspace_path.name} . --force\n"
                  f"    modal volume get introspection-vectors llama ./vectors_local --force\n"
                  f"or pass --subspace/--vectors-dir explicitly.")
    if angle_df is not None:
        draw_figure(df, judges, ks, alphas, angle_df,
                    out_path.with_stem(out_path.stem + "_angle"), run_label)

    # Per-judge, per-k table (rows: k, cols: alpha) so the numbers back the picture.
    for judge in judges:
        rate = spec_rate(df, judge)
        print(f"\n{spec_label(judge)} rate (rows: k, cols: alpha):")
        pivot = (rate.groupby([df["k"], df["alpha"]]).mean().unstack().round(2))
        print(pivot.to_string())
    if angle_df is not None:
        print("\nmean angle to subspace in degrees (rows: k, cols: alpha):")
        print(angle_df.groupby(["k", "alpha"])["angle"].mean().unstack().round(1).to_string())


if __name__ == "__main__":
    main()
