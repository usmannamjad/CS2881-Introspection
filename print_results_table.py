"""Print a summary table of every baseline and concept-vector experiment rate.

One row per run (or per config cell for the layer/coeff sweep), with the sample
count next to each rate. All positive rates are coherence-ANDed, matching the
plotting convention: a trial only counts as detected / identified if the response
was ALSO judged coherent, so "rambled incoherently and the judge pattern-matched"
never counts. NaN (ungraded) in any component judge drops the trial from that
rate's n, which is why n can differ slightly between columns of the same row.

The control run's identification column is dropped before computing (nothing was
injected, so 'correct identification' has no ground truth there), mirroring
plot_projection.py. Random-direction runs never grade identification at all.

    python print_results_table.py
"""
from pathlib import Path

import pandas as pd

RESULTS = Path("new_results")

DETECT = "affirmative_response"
IDENTIFY = "affirmative_response_followed_by_correct_identification"
COHERENCE = "coherence"


def to_rate(series):
    """Judge column values as 0/1 floats (NaN for ungraded), robust to CSV round-trips."""
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1.0: 1.0, 0.0: 0.0})


def and_rate(df, judges):
    """0/1 series for the AND of the judges; None if a column is missing/ungraded."""
    rate = None
    for judge in judges:
        col = f"{judge}_judge"
        if col not in df.columns:
            return None
        r = to_rate(df[col])
        rate = r if rate is None else rate * r
    rate = rate.dropna()
    return None if rate.empty else rate


def cell(df, judges):
    r = and_rate(df, judges)
    return "     --" if r is None else f"{r.mean() * 100:5.1f}%  (n={len(r):>4})"


def alpha_rows(path, alpha):
    df = pd.read_csv(path)
    return df[df["alpha"].notna() & (df["alpha"] == alpha)]


def main():
    control = pd.read_csv(RESULTS / "output_control_no_injection.csv")
    # Nothing injected -> no ground truth for identification; the column is
    # spuriously filled, so drop it (same as plot_projection.py).
    control = control.drop(columns=[f"{IDENTIFY}_judge"], errors="ignore")

    all_concepts = pd.read_csv(RESULTS / "output_all_concepts_layer15_coeff6.csv")
    reproduce = pd.read_csv(RESULTS / "output_anthropic_reproduce.csv")
    sweep = pd.read_csv(RESULTS / "output_coherence_affirmation.csv")

    stem = "projection_results_pca_subspace_all_concepts_layer15_coeff6"
    # Sweep baselines: the UNCENTERED alpha-sweep rows at alpha=0.5, pooled over k.
    # In an uncentered subspace, mixing proj and residual equally is exactly the
    # unprojected vector re-measured with k-times the trials (see plot_projection.py
    # --baseline-alpha), so these rows are the full concept / random vector.
    concept_alpha05 = alpha_rows(RESULTS / f"{stem}.csv", 0.5)
    random_alpha05 = pd.concat(
        [alpha_rows(RESULTS / f"{stem}_random.csv", 0.5),
         alpha_rows(RESULTS / f"{stem}_random_seed1.csv", 0.5)], ignore_index=True)

    rows = [
        ("BASELINES", None),
        ("no injection (control, L15)", control),
        ("random vectors (alpha-sweep seeds 0+1, alpha=0.5 rows)", random_alpha05),
        ("CONCEPT VECTORS", None),
        ("all concepts (L15, coeff 6, 300 concepts x 5 trials)", all_concepts),
        ("concepts identified >=1/5 (alpha-sweep, alpha=0.5 rows)", concept_alpha05),
        ("anthropic reproduce (50 concepts, L15, coeff 9)", reproduce),
        ("CONFIG SWEEP (mixed concepts, identification not graded)", None),
    ]
    for (layer, coeff), cell_df in sweep.groupby(["layer", "coeff"]):
        rows.append((f"  layer {layer}, coeff {coeff:g}", cell_df))

    name_w = max(len(name) for name, _ in rows) + 2
    header = ("experiment".ljust(name_w)
              + "coherent".center(18) + "coherent & detected".center(21)
              + "coherent & identified".center(21))
    print(header)
    print("-" * len(header))
    for name, df in rows:
        if df is None:
            print(name)
            continue
        print(name.ljust(name_w)
              + cell(df, [COHERENCE]).center(18)
              + cell(df, [COHERENCE, DETECT]).center(21)
              + cell(df, [COHERENCE, IDENTIFY]).center(21))


if __name__ == "__main__":
    main()
