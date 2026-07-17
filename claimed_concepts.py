"""What does the model CLAIM the injected thought is about, across conditions?

Free-text analysis of responses already on disk -- no model, no judge API calls.
Extracts the word each response claims the injected thought is about (the quoted
word in patterns like: the injected thought is about the word "coffee") and
compares the claim distributions of three conditions:

  control   no injection at all: the model's default confabulation vocabulary
  random    random norm-matched directions (projection_experiment.py --random,
            several seeds pooled): what the model claims when the injection is
            meaningless
  concept   real concept vectors (the projection sweep): claims with actual
            injected content

Two summary numbers per condition:
  * share of claims falling on the control run's top-N words -- how much of the
    "identification" is just the default prior (cloud, computer, dog, ...)
  * share of claims matching a TRAIN concept name from the PCA subspace fit --
    whether in-subspace directions pull claims toward the concepts the subspace
    encodes (for random alpha=0 rows this stays at baseline: the subspace raises
    DETECTION but its content is not read out)

Also prints, when the subspace .npz and the saved vectors are available, the nearest
concept vectors of each top principal component (cosine): the top PCs are semantic
CATEGORY directions (PC1 ~ countries/places, PC2 ~ famous historical figures), and
the in-subspace random claims at low k read those categories out -- at k=1 (where the
injected direction is literally +/-PC1) the top claim is "paris"/"capital", at k=5
"einstein" appears -- even though e.g. Paris is not one of the 300 concepts. So the
readout of meaningless in-subspace directions is categorical, not specific; the
per-k breakdown table shows it directly. (Interpretation caveats: at k=1, alpha=0
all draws collapse to just two distinct directions, +/-PC1; per-k claim counts are
only ~60-130.)

The claim extraction is a regex over quoted words; it misses unquoted claims, but
the miss rate is condition-independent, so shares are comparable across conditions.

    python claimed_concepts.py
    python claimed_concepts.py --random new_results/projection_results_..._random.csv \
        new_results/projection_results_..._random_seed1.csv

Writes plots/claimed_concepts.png (top claimed words per condition) and prints the
tables.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# the injected thought is about the word "coffee" / 'coffee' / smart quotes; sentence
# punctuation is often INSIDE the quotes ("capital.") and must not break the match
QUOTE_RE = re.compile(r'["“‘\']([A-Za-z][A-Za-z \-]{1,25})[.,;:!?]{0,2}["”’\']')
# quoted fragments that are not concept claims
NOT_CLAIMS = {"ok", "thought", "injected thought"}

DEFAULT_RANDOM = [
    "new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random.csv",
    "new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random_seed1.csv",
]


def claimed_words(responses):
    """All claimed words (lowercased) in a series of responses, one entry per claim."""
    words = []
    for r in responses.dropna():
        for m in QUOTE_RE.findall(str(r)):
            w = m.strip().lower()
            # <= 3 words and no 1-letter first token: concept names are short;
            # longer matches and leading single letters are sentence fragments
            # caught between two apostrophe-contractions ("m responding as if it",
            # "d say it" from I'm/I'd ... it's)
            parts = w.split()
            if w and w not in NOT_CLAIMS and len(parts) <= 3 and len(parts[0]) > 1:
                words.append(w)
    return words


def share(words, vocab):
    hits = sum(1 for w in words if w in vocab)
    return hits, len(words), (hits / len(words) if words else float("nan"))


def report_condition(label, words, control_vocab, train_names, top):
    counts = Counter(words)
    print(f"\n{label}: {len(words)} claims")
    print("  top:", ", ".join(f"{w} ({n})" for w, n in counts.most_common(top)))
    hits, tot, frac = share(words, control_vocab)
    print(f"  on control top-{len(control_vocab)} vocabulary: {hits}/{tot} = {frac:.0%}")
    if train_names:
        hits, tot, frac = share(words, train_names)
        print(f"  matching a TRAIN concept name:  {hits}/{tot} = {frac:.0%}")
    return counts


def pc_neighbors(subspace_path, vectors_dir, n_pcs=5, n_neighbors=5):
    """Nearest concept vectors of each top PC: reveals which semantic category each
    component encodes. Full table (all PCs, CSV export): pc_neighbors.py."""
    from pc_neighbors import pc_neighbor_table

    try:
        rows, n_vectors = pc_neighbor_table(subspace_path, vectors_dir, n_pcs, n_neighbors)
    except SystemExit as e:
        print(f"\n{e}; skipping the PC-neighbor listing.")
        return
    print(f"\nnearest concepts per top PC ({n_vectors} vectors, cosine, +/- direction):")
    for pc in sorted({r[0] for r in rows}):
        entries = [r for r in rows if r[0] == pc]
        print(f"  PC{pc} (median {entries[0][4]:+.2f}): "
              + ", ".join(f"{c} ({s:+.2f})" for _, _, c, s, _ in entries))


def main():
    parser = argparse.ArgumentParser(description="Compare claimed injected-thought words across conditions")
    parser.add_argument("--control", type=str, default="new_results/output_control_no_injection.csv",
                        help="No-injection control CSV (default: %(default)s)")
    parser.add_argument("--random", type=str, nargs="+", default=DEFAULT_RANDOM,
                        help="Random-direction projection CSVs; several seeds are pooled")
    parser.add_argument("--concept", type=str,
                        default="new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv",
                        help="Concept-vector projection CSV (default: %(default)s)")
    parser.add_argument("--subspace", type=str, default="pca_subspace_all_concepts_layer15_coeff6.npz",
                        help="pca_subspace_*.npz for the TRAIN concept names (skipped if missing)")
    parser.add_argument("--vectors-dir", type=str, default="./vectors_local/llama",
                        help="Saved {concept}_{layer}_{vec_type}.pt files, for the PC-neighbor "
                             "listing (skipped if missing)")
    parser.add_argument("--top", type=int, default=12, help="Top-N words to list per condition")
    parser.add_argument("--control-vocab-size", type=int, default=20,
                        help="Control top-N words used as the 'default prior' vocabulary")
    parser.add_argument("--out", type=str, default="plots/claimed_concepts.png", help="Output figure")
    parser.add_argument("--no_plot", action="store_true", help="Skip writing the figure")
    args = parser.parse_args()

    train_names = set()
    if Path(args.subspace).exists():
        d = np.load(args.subspace, allow_pickle=True)
        train_names = {str(w).lower() for w in d["train_names"]}
    else:
        print(f"Note: {args.subspace} not found; skipping the train-concept match rates.")

    ctrl_words = claimed_words(pd.read_csv(args.control)["response"])
    control_vocab = {w for w, _ in Counter(ctrl_words).most_common(args.control_vocab_size)}

    rand = pd.concat([pd.read_csv(p) for p in args.random], ignore_index=True)
    conc = pd.read_csv(args.concept)

    conditions = [
        ("control (no injection)", ctrl_words),
        ("random (all variants)", claimed_words(rand["response"])),
        ("random alpha=0 (in subspace)", claimed_words(rand.loc[rand["alpha"] == 0.0, "response"])),
        # alpha=0.5 mixes proj and residual back to (proj+residual)/2 = v/2, so its
        # DIRECTION equals the full random vector -- a consistency check on 'full'
        ("random alpha=0.5 (== full direction)",
         claimed_words(rand.loc[np.isclose(rand["alpha"].fillna(-1), 0.5), "response"])),
        ("random alpha=1 (orthogonal)", claimed_words(rand.loc[rand["alpha"] == 1.0, "response"])),
        ("concept vectors", claimed_words(conc["response"])),
    ]
    counts = {}
    for label, words in conditions:
        counts[label] = report_condition(label, words, control_vocab, train_names, args.top)

    # Train-concept match rate along the random sweep: flat => the subspace never
    # yields SPECIFIC concept readout (the categorical readout below is the signal).
    if train_names:
        print("\nrandom claims matching a TRAIN concept name, by alpha:")
        for a in sorted(rand.loc[rand["variant"] != "full", "alpha"].dropna().unique()):
            words = claimed_words(rand.loc[np.isclose(rand["alpha"].fillna(-1), a), "response"])
            hits, tot, frac = share(words, train_names)
            print(f"  alpha={a:.2f}: {hits:>3}/{tot:<4} = {frac:.0%}")

    # In-subspace claims per k: at low k the claims read out the CATEGORY of the top
    # PCs (paris/capital <- PC1 ~ places, einstein <- PC2 ~ famous people), and the
    # default-prior "cloud" share collapses vs alpha=1.
    print("\nrandom alpha=0 (in subspace) claims by k  [share of 'cloud' | top words]:")
    for k in sorted(rand.loc[rand["k"].notna(), "k"].unique()):
        c = Counter(claimed_words(rand.loc[(rand["alpha"] == 0.0) & (rand["k"] == k), "response"]))
        tot = sum(c.values())
        if not tot:
            continue
        print(f"  k={int(k):>2} ({tot:>3} claims, cloud {c['cloud'] / tot:>3.0%}): "
              + ", ".join(f"{w} ({n})" for w, n in c.most_common(6)))
    c1 = Counter(claimed_words(rand.loc[rand["alpha"] == 1.0, "response"]))
    if c1:
        print(f"  vs alpha=1, all k ({sum(c1.values())} claims): cloud {c1['cloud'] / sum(c1.values()):.0%}")

    # Which semantic category each top PC encodes (the source of the claims above).
    if Path(args.subspace).exists():
        pc_neighbors(args.subspace, args.vectors_dir)

    if not args.no_plot:
        # orthogonal random ~ control prior; in-subspace random shifts toward the top
        # PCs' semantic categories (places/people); concept vectors carry real content
        panels = ["control (no injection)", "random alpha=1 (orthogonal)",
                  "random alpha=0 (in subspace)", "concept vectors"]
        # sharex: the concept panel's flat, low bars vs the control/random "cloud"
        # spike IS the finding -- per-panel scales would visually erase it
        fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2), sharex=True)
        for ax, label in zip(axes, panels):
            c = counts[label]
            total = sum(c.values())
            top = c.most_common(10)[::-1]  # horizontal bars, biggest on top
            ys = range(len(top))
            ax.barh(ys, [n / total for _, n in top],
                    color=["#D55E00" if w in control_vocab else "#0072B2" for w, _ in top])
            ax.set_yticks(ys)
            ax.set_yticklabels([w for w, _ in top], fontsize=9)
            ax.set_xlabel("share of claims", fontsize=9)
            ax.set_title(f"{label}\n({total} claims)", fontsize=10)
            ax.grid(True, axis="x", alpha=0.3)
            ax.set_axisbelow(True)
        fig.suptitle("Claimed injected-thought words (orange = control run's default vocabulary)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = Path(args.out)
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
