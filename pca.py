"""PCA over the (L2-normed) steering vectors of *identified* concepts.

Goal: find a low-dimensional subspace that captures the concept vectors the model was
able to introspect on ("identified"), while holding out a test set of identified concepts
so a later projection experiment can check whether the subspace generalizes (rather than
just memorizing the concepts PCA was fit on).

Pipeline
--------
1. Read the results CSV and, per concept, count how many of its trials the
   `affirmative_response_followed_by_correct_identification` judge marked True (the
   "hit count", 0..n_trials). A concept counts as *identified* if hits >= --min-hits.
2. Load each identified concept's saved vector ({concept}_{layer}_{vec_type}.pt), and
   L2-normalize it -- the same normalization inject_concept_vector applies before steering.
3. Stratified train/test split by category (fixed seed) so every category is represented
   on both sides and PCA never sees the held-out concepts.
4. Fit PCA (SVD) on the TRAIN vectors only. Default is uncentered (subspace through the
   origin), which matches injecting vectors as directions; --center for conventional PCA.
5. Report explained variance and, as a first cheap generalization check, the reconstruction
   R^2 of TRAIN vs held-out TEST vectors under the top-k train subspace.
6. Save components, mean, singular values, and the exact train/test concept split so the
   projection experiment reuses the same held-out set.

    modal volume get introspection-vectors llama ./vectors_local --force  # get the .pt files first
    python pca.py --vectors-dir ./vectors_local/llama --layer 15

Writes pca_subspace_<run>.npz next to nothing in particular (see --out).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from plot_by_category import CATEGORY_KEYS, build_concept_category


def load_hits(csv_path, success_judge):
    """Per-concept hit count and category from the results CSV.

    Returns a DataFrame indexed by concept with columns [hits, n_trials, category].
    """
    df = pd.read_csv(csv_path)
    col = f"{success_judge}_judge"
    if col not in df:
        raise SystemExit(f"Column {col} not in {csv_path}; pass a valid --success-judge.")
    # Grading happens locally, so a CSV straight off the GPU has this column empty/'none'.
    # Track how many rows are actually graded (a real True/False) to tell "ungraded CSV"
    # apart from "graded but nothing met the threshold" downstream.
    is_graded = df[col].isin([True, False, "True", "False"])
    # True/False survive CSV round-trips as strings; map both to 1.0/0.0, NaN -> 0.
    hit = df[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).fillna(0.0)
    g = hit.groupby(df["concept"])
    out = pd.DataFrame({"hits": g.sum().astype(int), "n_trials": g.count().astype(int)})
    concept_to_category = build_concept_category("dataset/simple_data.json")
    out["category"] = out.index.map(concept_to_category)
    out.attrs["n_graded_rows"] = int(is_graded.sum())
    return out


def load_vectors(vectors_dir, concepts, layer, vec_type, normalize=True):
    """Load {concept}_{layer}_{vec_type}.pt for each concept; L2-normalize by default.

    Returns (X [n, d] float32, kept_names). Concepts whose file is missing are skipped
    (reported by the caller via the returned name list).
    """
    vectors_dir = Path(vectors_dir)
    if not vectors_dir.exists():
        raise SystemExit(f"Vectors dir {vectors_dir} does not exist. Pull it "
                         f"(modal volume get introspection-vectors llama <dir>) or fix --vectors-dir.")
    rows, names = [], []
    for concept in concepts:
        file_path = vectors_dir / f"{concept}_{layer}_{vec_type}.pt"
        if not file_path.exists():
            continue
        vec = torch.load(file_path, weights_only=False)["vector"]
        vec = np.asarray(vec.detach().cpu().float() if isinstance(vec, torch.Tensor) else vec,
                         dtype=np.float32).ravel()
        if normalize:
            n = np.linalg.norm(vec)
            if n == 0:
                continue
            vec = vec / n
        rows.append(vec)
        names.append(concept)
    if not rows:
        raise SystemExit(
            f"None of the {len(concepts)} identified concept(s) had a vector file in "
            f"{vectors_dir} (looked for *_{layer}_{vec_type}.pt). Check --layer/--vec-type "
            f"match the saved vectors, or that the dir holds this run's .pt files.")
    return np.vstack(rows), names


def stratified_split(names, categories, test_frac, seed):
    """Split concept indices into train/test, holding out ~test_frac of EACH category."""
    rng = np.random.default_rng(seed)
    names = np.asarray(names)
    categories = np.asarray(categories)
    train_idx, test_idx = [], []
    for cat in CATEGORY_KEYS:
        idx = np.where(categories == cat)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        # Keep at least one train sample per category even in tiny categories.
        n_test = min(n_test, len(idx) - 1) if len(idx) > 1 else 0
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def fit_pca(X_train, center):
    """SVD-based PCA on the training rows.

    Returns (mean [d], components [k, d] rows = principal axes, singular_values [k],
    explained_variance_ratio [k]). k = min(n_train, d). If center is False the mean is
    zeros and the subspace passes through the origin (the natural choice when the vectors
    are injected as directions).
    """
    mean = X_train.mean(axis=0) if center else np.zeros(X_train.shape[1], dtype=X_train.dtype)
    Xc = X_train - mean
    # full_matrices=False -> Vt is [k, d]; rows of Vt are the principal axes.
    # (U left singular vectors are per-sample coords; not needed here.)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total = (S ** 2).sum()
    evr = (S ** 2) / total if total > 0 else np.zeros_like(S)
    return mean, Vt, S, evr


def reconstruction_r2(X, mean, components, k):
    """Fraction of squared vector norm captured by projecting X onto the top-k axes.

    Uses the TRAIN mean/components so that on the test set this measures generalization.
    R^2 = 1 - ||X - recon||^2 / ||X - mean||^2, aggregated over all rows.
    """
    P = components[:k]                       # [k, d]
    Xc = X - mean
    recon = (Xc @ P.T) @ P                   # project then lift back
    ss_res = float(((Xc - recon) ** 2).sum())
    ss_tot = float((Xc ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description="PCA over normed identified-concept vectors")
    parser.add_argument("--csv", type=str,
                        default="new_results/output_all_concepts_layer15_coeff6.csv",
                        help="Results CSV used to decide which concepts were identified")
    parser.add_argument("--vectors-dir", type=str, default="./vectors_local/llama",
                        help="Directory of saved {concept}_{layer}_{vec_type}.pt files")
    parser.add_argument("--layer", type=int, default=15, help="Vector layer to load")
    parser.add_argument("--vec-type", type=str, default="avg", choices=["avg", "last"],
                        help="Which vector variant to load (default: avg, matches the run)")
    parser.add_argument("--success-judge", type=str,
                        default="affirmative_response_followed_by_correct_identification",
                        help="Judge whose True count defines a concept's hit count")
    parser.add_argument("--min-hits", type=int, default=1,
                        help="Concept is 'identified' if hits >= this (of n_trials). Default 1.")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fraction of each category held out for later validation")
    parser.add_argument("--center", action="store_true",
                        help="Mean-center before SVD (conventional PCA). Default: uncentered "
                             "subspace through the origin, matching direction injection. "
                             "Adds a _centered suffix to the default --out.")
    parser.add_argument("--seed", type=int, default=0, help="Split RNG seed")
    parser.add_argument("--out", type=str, default=None,
                        help="Output .npz (default: pca_subspace_<csv stem>.npz)")
    args = parser.parse_args()

    # 1. Identified concepts.
    hits = load_hits(args.csv, args.success_judge)
    n_trials = int(hits["n_trials"].max())
    identified = hits[(hits["hits"] >= args.min_hits) & hits["category"].notna()]
    if identified.empty:
        # Distinguish the two ways the identified set can be empty, so the user isn't sent
        # chasing the vectors (which are fine) by load_vectors' generic message.
        if hits.attrs.get("n_graded_rows", 0) == 0:
            raise SystemExit(
                f"No graded '{args.success_judge}' values in {args.csv}: the CSV looks "
                f"UNGRADED (that column is empty/'none'). Grading happens locally, so grade "
                f"it first:\n"
                f"    python judge_results.py --csv {args.csv} --judges {args.success_judge}\n"
                f"If pca runs on Modal, the container reads the CSV off the results Volume, so "
                f"upload the graded copy there too:\n"
                f"    modal volume put introspection-results {args.csv} {args.csv} --force")
        raise SystemExit(
            f"0 of {len(hits)} concepts reach --min-hits {args.min_hits} "
            f"(max hits observed = {int(hits['hits'].max())}). Lower --min-hits.")
    print(f"Identified concepts (hits >= {args.min_hits} of {n_trials}): "
          f"{len(identified)} / {len(hits)}")
    print(identified.groupby("category").size().reindex(CATEGORY_KEYS).fillna(0).astype(int)
          .to_string())

    # 2. Load + L2-normalize their vectors.
    X, names = load_vectors(args.vectors_dir, list(identified.index), args.layer, args.vec_type)
    missing = sorted(set(identified.index) - set(names))
    if missing:
        print(f"\nWarning: {len(missing)} identified concept(s) had no vector file, "
              f"skipped: {missing[:10]}")
    cats = [identified.loc[n, "category"] for n in names]
    print(f"Loaded {X.shape[0]} vectors of dim {X.shape[1]} (L2-normed).")

    # 3. Stratified held-out split (PCA never sees test).
    train_idx, test_idx = stratified_split(names, cats, args.test_frac, args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    train_names = [names[i] for i in train_idx]
    test_names = [names[i] for i in test_idx]
    print(f"\nTrain: {len(train_idx)}   Test (held out): {len(test_idx)}")

    # 4. Fit PCA on train only.
    mean, components, sing, evr = fit_pca(X_train, center=args.center)
    cum = np.cumsum(evr)
    print(f"\nPCA ({'centered' if args.center else 'uncentered / through origin'}) on train:")
    for thresh in (0.5, 0.9, 0.95, 0.99):
        k = int(np.searchsorted(cum, thresh) + 1)
        print(f"  {int(thresh*100)}% variance reached at k = {k} components")

    # 5. Generalization proxy: train vs held-out reconstruction R^2 by k.
    max_k = min(len(components), X.shape[1])
    ks = [k for k in (1, 2, 3, 5, 10, 20, 30, 50, max_k) if k <= max_k]
    ks = sorted(set(ks))
    print("\nReconstruction R^2 under top-k train subspace:")
    print("  k".rjust(5) + "train R^2".rjust(12) + "test R^2".rjust(12) + "cum.var".rjust(10))
    for k in ks:
        r2_tr = reconstruction_r2(X_train, mean, components, k)
        r2_te = reconstruction_r2(X_test, mean, components, k) if len(X_test) else float("nan")
        print(f"{k:5d}{r2_tr:12.3f}{r2_te:12.3f}{cum[k-1]:10.3f}")
    print("(test R^2 close to train R^2 => the subspace generalizes beyond the fitted concepts)")

    # 6. Save everything the projection experiment needs.
    run = Path(args.csv).stem.removeprefix("output_")
    if args.center:
        run += "_centered"   # keep centered fits from overwriting the uncentered outputs
    out_path = Path(args.out) if args.out else Path(f"pca_subspace_{run}.npz")
    np.savez(
        out_path,
        components=components, mean=mean, singular_values=sing,
        explained_variance_ratio=evr,
        train_names=np.array(train_names), test_names=np.array(test_names),
        train_categories=np.array([identified.loc[n, "category"] for n in train_names]),
        test_categories=np.array([identified.loc[n, "category"] for n in test_names]),
        layer=args.layer, vec_type=args.vec_type, centered=args.center,
        min_hits=args.min_hits, normalized=True,
    )
    print(f"\nSaved subspace + train/test split to {out_path}")
    print("Load later with: d = np.load(out_path, allow_pickle=False); "
          "d['components'], d['test_names'], ...")


if __name__ == "__main__":
    main()
