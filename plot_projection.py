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
sweep unequal angles (severely so for --random runs, where ~sqrt(k/4096) of the norm is in
the subspace and everything past alpha=0 bunches at 70-90 degrees -- prefer regenerating
with projection_experiment.py --sweep angle, whose variants are evenly spaced in angle;
such CSVs carry sweep='angle' and their alpha column is the sweep fraction t = angle/90). Injection re-normalizes every variant, but normalization only
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
                      concept-vector projection CSVs give the real-vector baseline,
                      labeled 'concepts identified >= 1/5 trials' because the sweeps
                      only run on concepts the all-concepts run identified in at least
                      1 of their 5 trials. (A random sweep's own built-in baseline is labeled
                      'random vector', since its 'full' rows inject the random
                      direction, not a concept.)
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

import matplotlib.patheffects as path_effects
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

# The projection sweeps only run on the test concepts that were identified in at least 1
# of their 5 trials in the full-injection all-concepts run, so the full-strength
# concept-vector baseline is named for that filtered population (vs the unfiltered
# "all concepts" line).
# mathtext \geq, not the unicode char: console prints of the label must survive
# Windows' cp1252 stdout encoding.
FULL_VECTOR_LABEL = r"concepts identified $\geq$ 1/5 trials"

# Okabe-Ito, colorblind-safe, assigned to k in fixed order (never cycled). Identity is
# always carried by the legend too, never color alone.
K_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
BASELINE_COLOR = "#555555"
# Reference lines are gray, telling them apart by dash pattern + legend, never color.
BASELINE_STYLES = {
    FULL_VECTOR_LABEL: ("#555555", (0, (4, 3))),
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


def draw_figure(df, judges, ks, alphas, angle_df, out_path, title, baselines=(),
                baseline_alpha=None, sweep="alpha", paper=False):
    """One panel per judge spec, one line per k. angle_df=None -> x is alpha; else x is the
    angle to the subspace in degrees, with each concept's rate scattered at its own angle.
    title: run description; single panel puts it in the panel title with the judge spec on
    the y axis, multi-panel puts it in the headline with the judge spec as panel subtitle.
    baselines: extra (style, label, dataframe) reference sources drawn as horizontal lines
    on every panel whose judge spec is graded in that dataframe (style keys BASELINE_STYLES).
    baseline_alpha: draw the run's own baseline from the sweep rows at this alpha (pooled
    over k) instead of the 'full' rows -- see the --baseline-alpha help for when the two
    are the same direction.
    paper: ICML style -- direct-label each baseline at the right end of its line (no rate
    number, no baseline legend box); at column width the gray dash patterns are no longer
    tellable apart in a legend, direct labels survive the shrink."""
    ncols = min(MAX_COLS, len(judges))
    nrows = ceil(len(judges) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.4 * nrows),
                             sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(judges):]:
        ax.set_visible(False)

    full_mask = df["variant"] == "full"

    # The run's own 'full' rows inject the concept vector -- except in a --random sweep,
    # where they inject the random direction, so the built-in baseline is named for what
    # it actually is.
    own_name = ("random vector"
                if "vector_source" in df.columns and (df["vector_source"] == "random").any()
                else FULL_VECTOR_LABEL)
    own_label = own_name
    if baseline_alpha is None:
        own_mask = full_mask
    else:
        # Every k injects the same direction at this alpha (for an uncentered subspace and
        # alpha=0.5 it IS the full vector), so pooling over k just multiplies the trials.
        # The alpha choice stays out of the label (it decorates every legend confusingly);
        # the console print and the README command document it.
        own_mask = df["alpha"].notna() & np.isclose(df["alpha"].fillna(-1.0), baseline_alpha)

    for panel, (ax, judge) in enumerate(zip(flat, judges)):
        rate = spec_rate(df, judge)
        # Reference lines: the run's own baseline rows, then any extra baseline runs.
        # Skipped where the spec isn't graded (mean of no rows = NaN). n stays out of the
        # legend (figure captions carry it); it is printed below instead.
        baseline_handles = []
        drawn_baselines = []
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
            line = ax.axhline(r.mean(), color=color, linestyle=dashes, linewidth=1.6,
                              label=f"{label} ({r.mean():.2f})", zorder=2)
            if paper:
                drawn_baselines.append((float(r.mean()), label, color))
            else:
                baseline_handles.append(line)
            print(f"[{spec_label(judge)}] baseline {label}: {r.mean():.3f} (n={len(r)})")

        k_handles = []
        curve_ends = {"left": [], "right": []}
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
                k_handles.append(
                    ax.errorbar(xs, means, yerr=errs, marker="o", markersize=7, linewidth=2,
                                capsize=3, color=color, label=f"k={k}", zorder=3))
                curve_ends["left"].append(means[0])
                curve_ends["right"].append(means[-1])

        # --paper: name each baseline at whichever end of its line the curves leave more
        # room, above it by default, below when another baseline sits close overhead but
        # not underfoot. A white stroke keeps the text readable over grid lines/curves.
        for y, label, color in drawn_baselines:
            clearance = {side: min((abs(y - v) for v in curve_ends[side]), default=np.inf)
                         for side in curve_ends}
            side = "right" if clearance["right"] >= clearance["left"] else "left"
            others = [o for o, _, _ in drawn_baselines if o != y]
            crowded_above = any(0 < o - y < 0.06 for o in others)
            crowded_below = any(0 < y - o < 0.06 for o in others)
            va = "top" if (crowded_above and not crowded_below) else "bottom"
            ax.text(0.985 if side == "right" else 0.015,
                    y + (-0.012 if va == "top" else 0.012), label,
                    transform=ax.get_yaxis_transform(),
                    ha=side, va=va, fontsize=8, color=color, zorder=4,
                    path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")])

        # Single panel: run description as panel title, metric on the y axis. Multi-panel:
        # run description once as the headline, metric as each panel's subtitle (the
        # judge differs per panel while sharey shares the scale, so the left column's
        # plain "Rate" label covers all panels).
        judge_label = spec_label(judge)
        metric = judge_label[0].upper() + judge_label[1:] + " rate"
        if len(judges) > 1:
            ax.set_title(metric, fontsize=12)
            if panel % ncols == 0:
                ax.set_ylabel("Rate", fontsize=11)
        else:
            ax.set_title(title, fontsize=12)
            ax.set_ylabel(metric, fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        # Anchor the ends of the sweep so proj/residual are unmistakable.
        if angle_df is None:
            if sweep == "angle":
                # Angle-sweep runs store the sweep fraction t in the alpha column; equal
                # steps are equal rotations (nominal angle = t*90 deg).
                ax.set_xlabel("sweep fraction t, uniform in angle "
                              "(0 = in subspace  ->  1 = orthogonal)", fontsize=10)
            else:
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
        # Two compact legends in the top corners instead of one big box over the data:
        # the coherence-ANDed rates top out well below 1.0, so that strip is free. Two
        # legends can't share matplotlib's "best" placement, hence the fixed corners.
        # (--paper direct-labels the baselines instead, so only the k legend remains.)
        if baseline_handles:
            base_legend = ax.legend(handles=baseline_handles, fontsize=8, loc="upper left",
                                    framealpha=0.9, title="baselines", title_fontsize=8)
            ax.add_artist(base_legend)
        ax.legend(handles=k_handles, fontsize=8, loc="upper right", framealpha=0.9,
                  title="subspace dim k", title_fontsize=8)

    if len(judges) > 1:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

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
                             "the full-strength concept-vector baseline (dashed), labeled as "
                             "the concepts identified at least once in the all-concepts run "
                             "(the population every sweep is restricted to). Meant for "
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
    parser.add_argument("--title", type=str, default=None,
                        help="Panel title describing the run (default: 'Random vector "
                             "projections' for --random sweeps, else 'Concept vector "
                             "projections')")
    parser.add_argument("--paper", action="store_true",
                        help="ICML style: direct-label each baseline at the right end of "
                             "its line (no rate numbers, no baseline legend box) and "
                             "write to ..._paper.png instead of overwriting the default")
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

    # Older CSVs predate the sweep column and are all alpha sweeps. In an angle sweep the
    # alpha column holds the sweep fraction t (nominal angle = t*90), so pooling the two
    # kinds would average incomparable x positions.
    sweep_modes = (set(df["sweep"].dropna().unique()) if "sweep" in df.columns else set())
    if len(sweep_modes) > 1:
        raise SystemExit(f"Refusing to pool CSVs with different sweep modes ({sweep_modes}): "
                         "their alpha columns mean different things.")
    sweep_mode = sweep_modes.pop() if sweep_modes else "alpha"

    own_baseline_alpha = args.baseline_alpha
    if args.baseline_alpha is not None and sweep_mode == "angle":
        # t=0.5 is the 45-degree direction, never the full vector, so the run's own
        # baseline falls back to its 'full' rows. External --full-csv/--random-csv sweeps
        # keep --baseline-alpha (it selects rows under THEIR alpha semantics).
        print("Note: angle-sweep CSV -- t=0.5 is the 45-degree direction, not the full "
              "vector; using the 'full' rows for the run's own baseline instead.")
        own_baseline_alpha = None

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
        # Row choice (like the run's own baseline) is documented in the console print and
        # the README command, not in the legend label.
        sel = src["alpha"].notna() & np.isclose(src["alpha"].fillna(-1.0), args.baseline_alpha)
        return style, style, src[sel]

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
        baselines.append(sweep_baseline(FULL_VECTOR_LABEL, args.full_csv))
    if args.random_csv:
        baselines.append(sweep_baseline("random vector", args.random_csv))

    is_random = "vector_source" in df.columns and (df["vector_source"] == "random").any()
    title = args.title or ("Random vector projections" if is_random
                           else "Concept vector projections")

    plots_dir = Path("plots")
    suffix = "_paper" if args.paper else ""
    out_path = (Path(args.out) if args.out
                else plots_dir / f"projection_curves_{run_label}{suffix}.png")
    draw_figure(df, judges, ks, alphas, None, out_path, title, baselines,
                own_baseline_alpha, sweep_mode, args.paper)

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
                    out_path.with_stem(out_path.stem + "_angle"), title, baselines,
                    own_baseline_alpha, sweep_mode, args.paper)

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
