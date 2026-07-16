"""Why does detection of random in-subspace directions depend on k? Geometry, no API.

Companion analysis to the --random projection runs (projection_experiment.py) and
claimed_concepts.py. Every injected random direction is reproducible from its seed
(default_rng([seed, concept_index]) over the held-out test concepts, exactly as
projection_experiment.py draws them), so this script reconstructs each draw's
alpha=0 direction and relates its GEOMETRY to the graded detection rate. Two parts:

1. k=1 sign split. Projecting any random vector onto a 1-D subspace gives +/-PC1 --
   all that survives of the 4096-d draw is the SIGN of v.PC1, so the k=1 cell is
   secretly just two injections. The concept clusters (countries etc.) sit on one
   side of PC1, making the two signs semantically opposite: toward concept content
   vs away from it. Detection splits dramatically (~0.50 toward vs ~0.13 away, vs
   ~0.39 full-random baseline) and only the toward-side claims are on-theme
   ("paris", "capital"): detection is SIGN-SENSITIVE, not a symmetric anomaly alarm.

2. Signed alignment vs detection, per k. Generalizes the sign split: correlate each
   draw's detection rate with the cosine of its injected direction to (a) the mean
   train-concept direction and (b) the nearest category centroid. Alignment predicts
   detection strongly at k=1-2 (r ~ 0.6-0.7), fades by k=5, and is gone at k=10-20
   even though detection is HIGHEST there (~0.6): at high k, ANY direction in the
   concept span is detected, with no readable content -- a subspace-level effect,
   not proximity to any single "general concept dimension".

Needs the graded random CSVs, the subspace .npz, the saved .pt vectors, and
dataset/simple_data.json (category centroids). Purely local, no model or judge.

    python random_direction_analysis.py
    python random_direction_analysis.py --random new_results/..._random.csv \
        new_results/..._random_seed1.csv
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

DETECT_JUDGE = "affirmative_response"
CATEGORY_KEYS = ["concept_vector_words", "famous_people", "countries",
                 "concrete_nouns", "abstract_nouns", "verbs"]
# sentence punctuation is often INSIDE the quotes ("capital.") -- allow it
QUOTE_RE = re.compile(r'["“‘\']([A-Za-z][A-Za-z \-]{1,25})[.,;:!?]{0,2}["”’\']')

DEFAULT_RANDOM = [
    "new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random.csv",
    "new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random_seed1.csv",
]


def to_rate(series):
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1.0: 1.0, 0.0: 0.0})


def top_claims(responses, n=8):
    words = []
    for r in responses.dropna():
        for m in QUOTE_RE.findall(str(r)):
            w = m.strip().lower()
            parts = w.split()
            # filter contraction fragments ("d say it" from I'd ... it's), like
            # claimed_concepts.claimed_words
            if w and w not in {"ok", "thought"} and len(parts) <= 3 and len(parts[0]) > 1:
                words.append(w)
    return Counter(words).most_common(n)


def load_unit_vectors(vectors_dir, layer, vec_type):
    unit = {}
    for fp in Path(vectors_dir).glob(f"*_{layer}_{vec_type}.pt"):
        v = torch.load(fp, weights_only=False)["vector"]
        v = np.asarray(v.detach().cpu().float() if isinstance(v, torch.Tensor) else v,
                       dtype=np.float32).ravel()
        n = np.linalg.norm(v)
        if n > 0:
            unit[fp.stem.removesuffix(f"_{layer}_{vec_type}")] = v / n
    return unit


def draw_direction(seed, concept_index, dim):
    """The random unit vector projection_experiment.py --random injects for this draw."""
    rng = np.random.default_rng([seed, concept_index])
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def main():
    parser = argparse.ArgumentParser(description="Relate random-direction geometry to detection")
    parser.add_argument("--random", type=str, nargs="+", default=DEFAULT_RANDOM,
                        help="Graded --random projection CSVs, in seed order")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="--random-seed of each CSV (default: 0 1 2 ... by position)")
    parser.add_argument("--subspace", type=str, default="pca_subspace_all_concepts_layer15_coeff6.npz")
    parser.add_argument("--vectors-dir", type=str, default="./vectors_local/llama")
    parser.add_argument("--dataset", type=str, default="dataset/simple_data.json",
                        help="Word lists for the category centroids")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(range(len(args.random)))
    if len(seeds) != len(args.random):
        raise SystemExit("--seeds must match --random one-to-one")

    d = np.load(args.subspace, allow_pickle=True)
    comps = d["components"].astype(np.float32)
    test_names = [str(w) for w in d["test_names"]]
    train_names = {str(w) for w in d["train_names"]}
    layer, vec_type = int(d["layer"]), str(d["vec_type"])
    concept_index = {c: i for i, c in enumerate(test_names)}

    unit = load_unit_vectors(args.vectors_dir, layer, vec_type)
    m = np.sum([unit[c] for c in train_names if c in unit], axis=0)
    m /= np.linalg.norm(m)
    cats = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    centroids = {}
    for key in CATEGORY_KEYS:
        vs = [unit[w] for w in cats[key] if w in unit]
        if vs:
            c = np.sum(vs, axis=0)
            centroids[key] = c / np.linalg.norm(c)

    frames = []
    for seed, path in zip(seeds, args.random):
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["det"] = to_rate(df[f"{DETECT_JUDGE}_judge"])
    full = df.loc[df["variant"] == "full", "det"].dropna()

    # ---- 1. k=1 sign split: the injected direction is sign(v.PC1) * PC1 ----
    pc1 = comps[0] / np.linalg.norm(comps[0])
    k1 = df[(df["alpha"] == 0.0) & (df["k"] == 1)].copy()
    k1["sign"] = [np.sign(float(draw_direction(s, concept_index[c], comps.shape[1]) @ pc1))
                  for s, c in zip(k1["seed"], k1["concept"])]
    # concept clusters sit on ONE side of PC1; orient labels from the mean direction
    toward = -1.0 if float(m @ pc1) < 0 else 1.0

    print("k=1 alpha=0: the injected direction is sign(v.PC1) * PC1, a coin flip per draw")
    print(f"full random baseline: {full.mean():.3f} ({len(full)} trials)\n")
    rates = {}
    for sgn, label in [(toward, "toward concept content"), (-toward, "away from it")]:
        sub = k1[k1["sign"] == sgn]
        r = sub["det"].dropna()
        rates[sgn] = r
        n_draws = sub.groupby(["seed", "concept"]).ngroups
        print(f"  {sgn:+.0f}*PC1 ({label}): {n_draws} draws, {len(r)} trials, detection {r.mean():.3f}")
        print(f"      claims: {top_claims(sub['response'])}")
    a, b = rates[toward], rates[-toward]
    p_pool = (a.sum() + b.sum()) / (len(a) + len(b))
    z = (a.mean() - b.mean()) / np.sqrt(p_pool * (1 - p_pool) * (1 / len(a) + 1 / len(b)))
    print(f"  toward vs away: z={z:+.2f}, two-sided p={2 * stats.norm.sf(abs(z)):.2e}")

    # ---- 2. signed alignment vs detection, per k ----
    det = df[df["alpha"] == 0.0].groupby(["seed", "concept", "k"])["det"].mean()
    ks = sorted(int(k) for k in df.loc[df["k"].notna(), "k"].unique())
    print("\nper-draw signed alignment of the injected direction vs detection:")
    print(f"  {'k':>3} {'mean cos(w,m)':>14} {'corr(det, cos_m)':>18} {'corr(det, best cat cos)':>25}")
    pooled_x, pooled_y = [], []
    for k in ks:
        P = comps[:k]
        xs_m, xs_cat, ys = [], [], []
        for seed in seeds:
            for c in test_names:
                if (seed, c, float(k)) not in det.index:
                    continue
                v = draw_direction(seed, concept_index[c], comps.shape[1])
                w = (v @ P.T) @ P
                w /= np.linalg.norm(w)
                xs_m.append(float(w @ m))
                xs_cat.append(max(float(w @ cen) for cen in centroids.values()))
                ys.append(det.loc[(seed, c, float(k))])
        xs_m, xs_cat, ys = map(np.array, (xs_m, xs_cat, ys))
        r_m, p_m = stats.pearsonr(xs_m, ys)
        r_c, p_c = stats.pearsonr(xs_cat, ys)
        print(f"  {k:>3} {xs_m.mean():>14.3f} {r_m:>+10.2f} (p={p_m:.3f}) {r_c:>+15.2f} (p={p_c:.4f})")
        pooled_x.append(xs_cat)
        pooled_y.append(ys)

    X, Y = np.concatenate(pooled_x), np.concatenate(pooled_y)
    r, p = stats.pearsonr(X, Y)
    print(f"\npooled across k ({len(X)} draws): corr(det, best-category cos) = {r:+.2f} (p={p:.2e})")
    print("Strong alignment-detection coupling at k=1-2 fading to none at k=10-20 (where "
          "detection is highest) => low k is direction-specific, high k is a subspace-level "
          "effect: any direction in the concept span gets detected, none get read out.")


if __name__ == "__main__":
    main()
