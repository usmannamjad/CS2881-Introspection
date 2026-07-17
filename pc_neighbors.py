"""Which concepts does each principal component of the PCA subspace point at?

For every stored PC of a pca_subspace_*.npz (pca.py), prints the concept vectors
with the largest |cosine| to it -- the table showing that the leading components
are semantic CATEGORY directions (PC1 ~ countries/places, PC2 ~ famous historical
figures), which is what the in-subspace random injections read out (see
claimed_concepts.py / random_direction_analysis.py).

Cosine = dot product of unit vectors = cos(angle); e.g. 0.89 is ~27 degrees, where
two random 4096-d directions sit at |cos| ~ 0.016 (~89 degrees). A PC's OVERALL
sign is an arbitrary PCA convention; only the relative signs within a component
are meaningful (clusters on one side vs the other, e.g. PC3 separates places from
gerunds). Note the fit is uncentered, so PC1 also tracks the shared mean direction
of the training vectors -- "mean concept vector, flavored toward places", not a
pure category axis.

    python pc_neighbors.py                      # top 5 neighbors of every stored PC
    python pc_neighbors.py --pcs 5              # just the table from the report
    python pc_neighbors.py --out pc_neighbors.csv   # full result as CSV

Purely local (subspace .npz + saved .pt vectors), no model or API.
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def load_unit_vectors(vectors_dir, layer, vec_type):
    """{concept: unit vector} for every saved {concept}_{layer}_{vec_type}.pt."""
    unit = {}
    for fp in Path(vectors_dir).glob(f"*_{layer}_{vec_type}.pt"):
        v = torch.load(fp, weights_only=False)["vector"]
        v = np.asarray(v.detach().cpu().float() if isinstance(v, torch.Tensor) else v,
                       dtype=np.float32).ravel()
        n = np.linalg.norm(v)
        if n > 0:
            unit[fp.stem.removesuffix(f"_{layer}_{vec_type}")] = v / n
    return unit


def pc_neighbor_table(subspace_path, vectors_dir, n_pcs=None, n_neighbors=5):
    """Rows of (pc_index, rank, concept, cosine, pc_median): the n_neighbors concepts
    with the largest |cosine| to each of the first n_pcs components (default: all
    stored). pc_median is the median cosine of ALL vectors to that PC -- the
    distribution's center, showing e.g. that PC1 is shifted wholesale (median -0.56),
    not just at its top neighbors."""
    d = np.load(subspace_path, allow_pickle=True)
    comps = d["components"].astype(np.float32)
    layer, vec_type = int(d["layer"]), str(d["vec_type"])

    unit = load_unit_vectors(vectors_dir, layer, vec_type)
    if not unit:
        raise SystemExit(f"No vectors found in {vectors_dir}")
    names = list(unit)
    V = np.stack([unit[c] for c in names])

    rows = []
    for i in range(min(n_pcs or comps.shape[0], comps.shape[0])):
        p = comps[i] / np.linalg.norm(comps[i])
        sims = V @ p
        med = float(np.median(sims))
        for rank, j in enumerate(np.argsort(-np.abs(sims))[:n_neighbors], start=1):
            rows.append((i + 1, rank, names[j], float(sims[j]), med))
    return rows, len(names)


def main():
    parser = argparse.ArgumentParser(description="Nearest concept vectors of each PCA component")
    parser.add_argument("--subspace", type=str, default="pca_subspace_all_concepts_layer15_coeff6.npz",
                        help="pca_subspace_*.npz written by pca.py (default: %(default)s)")
    parser.add_argument("--vectors-dir", type=str, default="./vectors_local/llama",
                        help="Saved {concept}_{layer}_{vec_type}.pt files")
    parser.add_argument("--pcs", type=int, default=None,
                        help="How many leading PCs to list (default: all stored)")
    parser.add_argument("--neighbors", type=int, default=5,
                        help="Concepts to list per PC (default: 5)")
    parser.add_argument("--out", type=str, default=None,
                        help="Also write the table as CSV (pc, rank, concept, cosine)")
    args = parser.parse_args()

    rows, n_vectors = pc_neighbor_table(args.subspace, args.vectors_dir, args.pcs, args.neighbors)

    print(f"nearest concepts per PC of {args.subspace} "
          f"({n_vectors} vectors, cosine, +/- direction; median = center of the "
          f"full cosine distribution to that PC):")
    for pc in sorted({r[0] for r in rows}):
        entries = [r for r in rows if r[0] == pc]
        med = entries[0][4]
        print(f"  PC{pc} (median {med:+.2f}): "
              + ", ".join(f"{c} ({s:+.2f})" for _, _, c, s, _ in entries))

    if args.out:
        import pandas as pd
        pd.DataFrame(rows, columns=["pc", "rank", "concept", "cosine", "pc_median"]) \
            .to_csv(args.out, index=False)
        print(f"\nTable saved to {args.out}")


if __name__ == "__main__":
    main()
