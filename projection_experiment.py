"""Test whether the PCA subspace (from pca.py) is useful for injection.

Idea: pca.py fit a k-dim subspace on the TRAIN identified concepts and held out a TEST
set it never saw. Here we inject, for each held-out concept, a sweep of directions that
interpolates from "entirely inside the subspace" to "entirely orthogonal to it", and let
the same judges grade each one. For each k we split v into two orthogonal parts,
v = proj + residual (proj in the subspace, residual its orthogonal complement), and mix:

    d(alpha) = (1 - alpha) * proj + alpha * residual

  proj_k        alpha=0    : entirely inside the top-k subspace
  interp_k_aXX  0<alpha<1  : --interp-steps evenly-spaced partial mixes (default 5)
  residual_k    alpha=1    : entirely orthogonal to the subspace (control)
  full                     : the raw L2-normed vector (baseline; == alpha=0.5 direction,
                             since (proj+residual)/2 is parallel to v -- a built-in check)

Interpretation: sweep the identification rate along alpha. If it stays high near alpha=0
and falls off toward alpha=1, identifiability lives in the subspace (the useful result);
a flat curve means the subspace carries no special information.

Injection re-normalizes every vector, so only the DIRECTION of each variant matters; the
mix's changing magnitude is irrelevant -- alpha only rotates the injected direction from
proj toward residual. Reuses main.test_vector_multiple_choice unchanged by writing each
variant to a temp {concept}_{layer}_{vec_type}.pt file.

Like the other GPU steps, this GENERATES ONLY by default (--judges none): no OpenAI calls
run on the GPU. It writes projection_results_<subspace stem>.csv (judge columns empty) plus
the judge_question_<...>.txt the graders need, so grading happens locally afterwards -- the
GPU isn't billed while it waits on the judge API.

Run where the model lives. On Modal: `modal run modal_app.py::run_pca` then
`modal run modal_app.py::run_projection`, then download + grade locally:

    # pull the CSV + judge_question file (they sit at the volume root; name them explicitly)
    modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6.csv . --force
    modal volume get introspection-results judge_question_projection_results_pca_subspace_all_concepts_layer15_coeff6.txt . --force
    python judge_results.py --csv projection_results_pca_subspace_all_concepts_layer15_coeff6.csv \
        --judges coherence affirmative_response affirmative_response_followed_by_correct_identification
    python plot_projection.py --csv projection_results_pca_subspace_all_concepts_layer15_coeff6.csv

Locally end-to-end (needs a GPU for generation, OPENAI_API_KEY for the grading step):

    python projection_experiment.py --subspace pca_subspace_all_concepts_layer15_coeff6.npz \
        --vectors-dir ./vectors_local/llama --ks 5 10 20 --interp-steps 5 \
        --coeff 6 --temperature 0.8 --trials 5
    python judge_results.py --csv projection_results_pca_subspace_all_concepts_layer15_coeff6.csv \
        --judges coherence affirmative_response affirmative_response_followed_by_correct_identification
    python plot_projection.py --csv projection_results_pca_subspace_all_concepts_layer15_coeff6.csv

(Pass --judges to projection_experiment.py to grade inline instead, e.g. for a quick local test.)
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from all_prompts import get_anthropic_reproduce_messages
from main import test_vector_multiple_choice

# Judge whose True count = "identified"; matches pca.py / plot_by_category.
SUCCESS_JUDGE = "affirmative_response_followed_by_correct_identification"
# Detection ("noticed an injected thought" without necessarily naming it). Weaker than
# identification, so it should persist further along the proj->orthogonal sweep.
DETECT_JUDGE = "affirmative_response"
# Judges graded/plotted by default for the projection sweep, weakest signal first.
CURVE_JUDGES = [DETECT_JUDGE, SUCCESS_JUDGE]


def project(v, mean, components, k):
    """Project v onto the top-k train subspace. Matches pca.py's centering convention:
    mean is zeros for an uncentered (through-origin) fit, so proj = (v @ P.T) @ P there."""
    P = components[:k]                       # [k, d]
    return mean + (v - mean) @ P.T @ P


def build_variants(v, mean, components, ks, interp_steps, normalize=True):
    """Directions to inject for one concept vector v."""

    target_norm = np.linalg.norm(v)

    def normalize_vec(vec):
        if not normalize:
            return vec

        norm = np.linalg.norm(vec)
        if np.isclose(norm, 0.0):
            return None

        # Use `vec / norm` here instead for unit vectors.
        # Normalizing to targert norm so coefficents used in previous experiments match
        return vec / norm * target_norm

    out = []

    full = normalize_vec(v)
    if full is not None:
        out.append(("full", full, None, float("nan")))

    alphas = np.linspace(0.0, 1.0, interp_steps + 2)

    for k in ks:
        proj = project(v, mean, components, k)
        residual = v - proj

        for a in alphas:
            vec = (1.0 - a) * proj + a * residual
            vec = normalize_vec(vec)

            if vec is None:
                continue

            if np.isclose(a, 0.0):
                name = f"proj_{k}"
            elif np.isclose(a, 1.0):
                name = f"residual_{k}"
            else:
                name = f"interp_{k}_a{int(round(a * 100)):02d}"

            out.append((name, vec, k, float(a)))

    return out


def main():
    parser = argparse.ArgumentParser(description="Inject PCA-projected vectors on held-out concepts")
    parser.add_argument("--subspace", type=str, required=True,
                        help="pca_subspace_*.npz written by pca.py")
    parser.add_argument("--vectors-dir", type=str, default="./vectors_local/llama",
                        help="Directory of saved {concept}_{layer}_{vec_type}.pt files")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20],
                        help="Subspace dimensions to test (each gets a proj_k and residual_k)")
    parser.add_argument("--interp-steps", type=int, default=5,
                        help="Interior mixes between proj (in-subspace) and residual "
                             "(orthogonal), per k. Default 5.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train", "all"],
                        help="Which concepts to inject (default: the held-out test set)")
    parser.add_argument("--coeff", type=float, default=6.0, help="Injection coefficient")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (matches the original run's 0.8)")
    parser.add_argument("--trials", type=int, default=10, help="Generations per (concept, variant)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--judges", type=str, nargs="+", default=["none"],
                        help="Judges to run INLINE. Default 'none' = generate only (no OpenAI "
                             "calls on the GPU); grade later with judge_results.py. Pass e.g. "
                             "'coherence affirmative_response_followed_by_correct_identification' "
                             "to judge inline instead.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV (default: projection_results_<subspace stem>.csv)")
    args = parser.parse_args()

    # 1. Load subspace + the exact split pca.py held out.
    d = np.load(args.subspace, allow_pickle=True)
    components = d["components"].astype(np.float32)
    mean = d["mean"].astype(np.float32)
    layer = int(d["layer"])
    vec_type = str(d["vec_type"])
    max_k = components.shape[0]
    ks = [k for k in args.ks if k <= max_k] or [max_k]
    if any(k > max_k for k in args.ks):
        print(f"Note: capping k at {max_k} (rank of the fitted subspace).")

    if args.split == "train":
        names = list(d["train_names"])
    elif args.split == "all":
        names = list(d["train_names"]) + list(d["test_names"])
    else:
        names = list(d["test_names"])
    print(f"Injecting {len(names)} {args.split} concept(s) at layer {layer}, coeff {args.coeff}, "
          f"vec_type {vec_type}; ks={ks}")

    # 2. Load the model once and reuse across every injection.
    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # "none" (the default) => generate only, no OpenAI calls on the GPU; grade afterwards
    # locally with judge_results.py, matching main.py / run_experiment.
    judges_to_run = [] if "none" in args.judges else args.judges
    if not judges_to_run:
        print("Judging OFF (generate only) -- grade the CSV locally with judge_results.py.")

    vectors_dir = Path(args.vectors_dir)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for concept in names:
            fp = vectors_dir / f"{concept}_{layer}_{vec_type}.pt"
            if not fp.exists():
                print(f"  missing vector, skipped: {concept}")
                continue
            raw = torch.load(fp, weights_only=False)["vector"]
            v = np.asarray(raw.detach().cpu().float() if isinstance(raw, torch.Tensor) else raw,
                           dtype=np.float32).ravel()
            n = np.linalg.norm(v)
            if n == 0:
                continue
            v = v / n                        # same L2-norm the injector applies

            for variant, vec, k, alpha in build_variants(v, mean, components, ks, args.interp_steps):
                # Reuse test_vector_multiple_choice unchanged: it reads {'vector'} from a
                # file and parses concept/layer/vec_type from the filename, so keep that name.
                tmp_fp = tmp / f"{concept}_{layer}_{vec_type}.pt"
                torch.save({"vector": torch.tensor(vec, dtype=torch.float32),
                            "concept_name": concept, "layer": layer, "vec_type": vec_type,
                            "model_name": args.model}, tmp_fp)
                results = test_vector_multiple_choice(
                    str(tmp_fp), model=model, tokenizer=tokenizer, type="anthropic_reproduce",
                    coeff=args.coeff, judges=judges_to_run, temperature=args.temperature,
                    num_samples=args.trials,
                )
                for trial_i, r in enumerate(results):
                    r["variant"] = variant
                    r["k"] = k
                    r["alpha"] = alpha
                    r["trial"] = trial_i
                    r["temperature"] = args.temperature   # for judge_results.py plot compat
                    rows.append(r)

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path(f"projection_results_{Path(args.subspace).stem}.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")

    # Write the conversation text the coherence/affirmative judges grade against, named the
    # way judge_results.py expects (judge_question_<run_label>.txt next to the CSV, where
    # run_label = csv stem minus 'output_'). Every row of an anthropic_reproduce run shares
    # this identical prompt, so one file suffices -- this is what lets grading happen later.
    run_label = out.stem.removeprefix("output_")
    judge_question = tokenizer.apply_chat_template(
        get_anthropic_reproduce_messages(), tokenize=False, add_generation_prompt=False)
    (out.parent / f"judge_question_{run_label}.txt").write_text(judge_question, encoding="utf-8")

    # 3. Interpolation curves per judge, per k (only if the judges ran inline). Detection
    #    (weaker) then identification: detection should stay higher along the sweep.
    graded = [j for j in CURVE_JUDGES if f"{j}_judge" in df and df[f"{j}_judge"].notna().any()]
    if graded:
        for judge in graded:
            rate = df[f"{judge}_judge"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
            full_rate = rate[df["variant"] == "full"].mean()
            print(f"\n{judge} rate along alpha (0 = in-subspace proj -> 1 = orthogonal residual)"
                  f"   [full baseline = {full_rate:.3f}]:")
            for k in ks:
                mask = df["k"] == k
                if not mask.any():
                    continue
                g = rate[mask].groupby(df.loc[mask, "alpha"]).mean().sort_index()
                cells = "  ".join(f"a{a:.2f}={m:.2f}" for a, m in g.items())
                print(f"  k={k:<3d} {cells}")
        print("\nHigh near a0 falling toward a1 => the signal lives in the subspace; "
              "detection typically persists further than identification.")
        print(f"Plot the curves with: python plot_projection.py --csv {out}")
    else:
        print(f"\nGenerate-only run. Grade locally, then plot the curves:\n"
              f"    python judge_results.py --csv {out} \\\n"
              f"        --judges coherence {DETECT_JUDGE} {SUCCESS_JUDGE}\n"
              f"    python plot_projection.py --csv {out}")


if __name__ == "__main__":
    main()
