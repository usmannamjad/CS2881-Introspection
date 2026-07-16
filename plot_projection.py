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

--csv accepts several files of the same sweep and pools their trials -- meant for --random
replicates run at different --random-seed values (seed 0 = ..._random.csv, seed N =
..._random_seed<N>.csv). Pooled outputs get a _pooled<N> suffix:

    python plot_projection.py --csv new_results/projection_results_..._random.csv \
        new_results/projection_results_..._random_seed1.csv

Two optional reference lines put the curves in context (drawn on every panel whose judge
they can be computed for):

  --no-injection-csv  a graded control run with no vector injected (e.g.
                      new_results/output_control_no_injection.csv). Its rate is the
                      judge's false-positive floor -- e.g. the model claims to detect a
                      thought ~19% of the time when nothing was injected.
  --random-csv        graded --random projection CSVs (several seeds are pooled). Only
                      their 'full' rows are used: the rate when a random Gaussian
                      direction (norm-matched to the concept vector) is injected at full
                      strength -- "any strong-enough perturbation trips the detector"
                      baseline. Judges that are ungraded there (e.g. correct
                      identification, which has no ground truth for a random direction)
                      are skipped automatically.
  --full-csv          the mirror image, for plotting a --random sweep: 'full' rows of the
                      concept-vector projection CSVs give the real-vector baseline. (A
                      random sweep's own built-in baseline is labeled 'random vector',
                      since its 'full' rows inject the random direction, not a concept.)
  --all-concepts-csv  a graded full-injection output_*.csv over the whole concept set
                      (e.g. new_results/output_all_concepts_layer15_coeff6.csv): the
                      unselected population rate, next to which the sweep's baseline
                      shows how much the identified-at-least-once filter inflates it.

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
    f"{COHERENCE_JUDGE}+{SUCCESS_JUDGE}": "coherent & correctly identified",
}

# Okabe-Ito, colorblind-safe, assigned to k in fixed order (never cycled). Identity is
# always carried by the legend too, never color alone.
K_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
BASELINE_COLOR = "#555555"
# Reference lines are gray, telling them apart by dash pattern + legend, never color.
BASELINE_STYLES = {
    "full vector": ("#555555", (0, (4, 3))),
    "no injection": ("#999999", (0, (1, 1.8))),
    "random vector": ("#333333", (0, (6, 1.5, 1, 1.5))),
    "all concepts": ("#777777", (0, (2.5, 2.5))),
}
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


def draw_figure(df, judges, ks, alphas, angle_df, out_path, run_label, baselines=(),
                baseline_alpha=None):
    """One panel per judge spec, one line per k. angle_df=None -> x is alpha; else x is the
    angle to the subspace in degrees, with each concept's rate scattered at its own angle.
    baselines: extra (style, label, dataframe) reference sources drawn as horizontal lines
    on every panel whose judge spec is graded in that dataframe (style keys BASELINE_STYLES).
    baseline_alpha: draw the run's own baseline from the sweep rows at this alpha (pooled
    over k) instead of the 'full' rows -- see the --baseline-alpha help for when the two
    are the same direction."""
    ncols = min(MAX_COLS, len(judges))
    nrows = ceil(len(judges) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.4 * nrows),
                             sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(judges):]:
        ax.set_visible(False)

    full_mask = df["variant"] == "full"
    full_n = int(spec_rate(df, judges[0])[full_mask].notna().sum())

    # The run's own 'full' rows inject the concept vector -- except in a --random sweep,
    # where they inject the random direction, so the built-in baseline is named for what
    # it actually is.
    own_name = ("random vector"
                if "vector_source" in df.columns and (df["vector_source"] == "random").any()
                else "full vector")
    if baseline_alpha is None:
        own_mask, own_label = full_mask, own_name
    else:
        # Every k injects the same direction at this alpha (for an uncentered subspace and
        # alpha=0.5 it IS the full vector), so pooling over k just multiplies the trials.
        own_mask = df["alpha"].notna() & np.isclose(df["alpha"].fillna(-1.0), baseline_alpha)
        own_label = f"{own_name} @ alpha={baseline_alpha:g}"

    for ax, judge in zip(flat, judges):
        rate = spec_rate(df, judge)
        # Reference lines: the run's own baseline rows, then any extra baseline runs.
        # Skipped where the spec isn't graded (mean of no rows = NaN).
        for name, label, src, src_rate in ([(own_name, own_label, df, rate[own_mask])]
                                           + [(s, l, b, None) for s, l, b in baselines]):
            if src_rate is None:
                if any(f"{p}_judge" not in src.columns for p in judge.split("+")):
                    continue
                src_rate = spec_rate(src, judge)
            r = src_rate.dropna()
            if r.empty:
                continue
            color, dashes = BASELINE_STYLES[name]
            ax.axhline(r.mean(), color=color, linestyle=dashes, linewidth=1.6,
                       label=f"{label} ({r.mean():.2f}, n={len(r)})", zorder=2)

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
    parser.add_argument("--csv", type=str, nargs="+", required=True,
                        help="Graded projection_results_*.csv. Several CSVs of the same sweep "
                             "(e.g. --random runs at different --random-seed values) are "
                             "pooled; the output gets a _pooled<N> suffix so single-run "
                             "plots aren't overwritten")
    parser.add_argument("--judges", type=str, nargs="+", default=DEFAULT_JUDGES,
                        help="Judge columns to plot, one subplot each; combine with '+' for "
                             "logical AND, e.g. coherence+affirmative_response "
                             "(default: detection, identification, coherence, and both ANDs)")
    parser.add_argument("--no-injection-csv", type=str, default=None,
                        help="Graded control CSV with no vector injected (e.g. "
                             "new_results/output_control_no_injection.csv); drawn as a "
                             "dotted false-positive-floor line on every applicable panel")
    parser.add_argument("--random-csv", type=str, nargs="+", default=None,
                        help="Graded --random projection CSVs (seeds are pooled); their "
                             "'full' rows give the full-strength random-direction baseline, "
                             "drawn as a dash-dot line on every applicable panel")
    parser.add_argument("--baseline-alpha", type=float, default=None,
                        help="Draw the run's own baseline from the sweep rows at this alpha "
                             "(pooled over k) instead of the 'full' rows. alpha=0.5 mixes "
                             "proj and residual equally, which for an UNCENTERED subspace "
                             "is exactly the full-vector direction re-measured inside the "
                             "sweep with k-times the trials; for a centered subspace the "
                             "direction differs, so there it is just 'the mid-sweep rate'")
    parser.add_argument("--full-csv", type=str, nargs="+", default=None,
                        help="Graded concept-vector projection CSVs; their 'full' rows give "
                             "the full-strength concept-vector baseline (dashed). Meant for "
                             "plotting a --random sweep next to the real-vector rate")
    parser.add_argument("--all-concepts-csv", type=str, default=None,
                        help="Graded full-injection output_*.csv over the whole concept set; "
                             "its rate is the unselected-population baseline (the sweep only "
                             "uses concepts that were identified at least once)")
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

    csv_paths = [Path(p) for p in args.csv]
    csv_path = csv_paths[0]
    # base_label (first CSV) also drives the subspace-.npz lookup below; the pooled
    # suffix only decorates outputs, so plots of a single run are never overwritten.
    base_label = csv_path.stem.removeprefix("projection_results_").removeprefix("output_")
    run_label = base_label + (f"_pooled{len(csv_paths)}" if len(csv_paths) > 1 else "")
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    if len(csv_paths) > 1:
        print(f"Pooling {len(csv_paths)} CSVs ({len(df)} rows total).")

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

    def sweep_baseline(style, paths):
        """External sweep CSVs as a baseline source, using the same rows the run's own
        baseline uses: 'full' rows, or the sweep rows at --baseline-alpha (pooled over k;
        for an uncentered subspace and alpha=0.5 that is the same direction, re-measured
        with k-times the trials)."""
        src = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        if args.baseline_alpha is None:
            return style, style, src[src["variant"] == "full"]
        sel = src["alpha"].notna() & np.isclose(src["alpha"].fillna(-1.0), args.baseline_alpha)
        return style, f"{style} @ alpha={args.baseline_alpha:g}", src[sel]

    baselines = []
    if args.no_injection_csv:
        ctrl = pd.read_csv(args.no_injection_csv)
        # Nothing is injected in the control run, so 'correct identification' has no ground
        # truth there and was never actually measured -- drop the spuriously-filled column
        # so no fake 0% line lands on identification panels.
        ctrl = ctrl.drop(columns=[f"{SUCCESS_JUDGE}_judge"], errors="ignore")
        baselines.append(("no injection", "no injection", ctrl))
    if args.all_concepts_csv:
        baselines.append(("all concepts", "all concepts", pd.read_csv(args.all_concepts_csv)))
    if args.full_csv:
        baselines.append(sweep_baseline("full vector", args.full_csv))
    if args.random_csv:
        baselines.append(sweep_baseline("random vector", args.random_csv))

    plots_dir = Path("plots")
    out_path = Path(args.out) if args.out else plots_dir / f"projection_curves_{run_label}.png"
    draw_figure(df, judges, ks, alphas, None, out_path, run_label, baselines,
                args.baseline_alpha)

    # Angle-axis version: same curves, x = actual degrees between direction and subspace.
    # Newer CSVs store the exact per-row angle, so no subspace/.pt files are needed -- and
    # it is the only correct source for --random runs, whose injected directions aren't on
    # disk (the .pt files hold the CONCEPT vectors, so recomputing would give wrong angles).
    if "angle" in df.columns and df["angle"].notna().any():
        # Mean (not drop_duplicates): pooled random seeds inject a different direction per
        # seed, so the same (concept, k, alpha) carries several angles -- average them for
        # the curve's x position. For a single CSV the angle is constant per group anyway.
        angle_df = (df.loc[df["variant"] != "full", ["concept", "k", "alpha", "angle"]]
                    .dropna(subset=["angle"])
                    .groupby(["concept", "k", "alpha"], as_index=False)["angle"].mean())
    elif "vector_source" in df.columns and (df["vector_source"] == "random").any():
        print("Random-direction run without a stored angle column: skipping the angle-axis "
              "figure (vector files hold the concept vectors, not the injected random "
              "directions).")
        angle_df = None
    else:
        if args.subspace:
            subspace_path = Path(args.subspace)
        else:
            name = base_label if base_label.startswith("pca_subspace") else f"pca_subspace_{base_label}"
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
                    out_path.with_stem(out_path.stem + "_angle"), run_label, baselines,
                    args.baseline_alpha)

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
