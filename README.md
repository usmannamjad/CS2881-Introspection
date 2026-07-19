# CS2881 Introspection

Injection-detection experiments on Llama-3.1-8B-Instruct. Simple data uses 50 mixed concepts + 250 
category-specific concept words from Jack Lindsey's "Emergent Introspective Awareness in Large 
Language Models" 2025 paper. Overlapping or near overlapping concepts were replaced with new ones.

`old_results` and `old_plots` contain previous results by Hahami et al., `new_results`
and `plots` have our results.

Added the examples to the coherence judge prompt since a significant number of coherent
responses were judged incoherent in the control experiment.

## Setup

GPU generation runs on Modal (A10G); judging runs locally afterwards so no GPU time is
wasted waiting on OpenAI. One-time setup:

```bash
pip install modal
modal setup
modal secret create openai-secret OPENAI_API_KEY=sk-...
modal secret create huggingface-secret HF_TOKEN=hf_...
```

The commands below are the Modal versions that produced `new_results/`. Each entrypoint
is a thin wrapper around a local script (`save_vectors.py`, `main.py`, `pca.py`,
`projection_experiment.py`) with the same flags — see the top of `modal_app.py` for the
mapping — so everything also runs fully locally on a ~16 GB GPU, e.g.:

```bash
python save_vectors.py --save_dir ./saved_vectors/llama
python main.py --type anthropic_reproduce --layers 15 --coeffs 9 \
  --vectors_dir ./saved_vectors/llama --vec_types avg
```

Judging is always local and resumable (`judge_results.py` only fills empty judge
columns, so re-running after an API hiccup is safe). It needs `OPENAI_API_KEY` set.

## 1. Concept vectors

```bash
# Layer-15 avg-activation vectors for all 300 concepts (50 main concept_vector_words
# + 5 categories x 50). Already-saved vectors are skipped:
modal run modal_app.py::all_concepts --step vectors
```

## 2. Baselines

```bash
# No-injection control: 100 generations of the anthropic_reproduce prompt at coeff=0,
# temp 0.8 -> new_results/output_control_no_injection.csv. This is the false-positive
# floor ("no injection" line in the figures). The identification judge has no ground
# truth here (nothing is injected), so only coherence + detection are judged:
modal run modal_app.py::control
modal volume get introspection-results new_results . --force
python judge_results.py --csv new_results/output_control_no_injection.csv --judges coherence affirmative_response
```

The random-vector baseline is not a separate run: it is the alpha=0.5 rows (= the
unprojected random directions, since proj + residual mixed equally reconstruct the full
vector in the uncentered subspace) of the random alpha sweeps in step 5, pooled over k
and seeds. Likewise the "concepts identified >= 1/5 trials" baseline is the alpha=0.5
rows of the concept alpha sweep.

## 3. All-concepts (category) experiment

```bash
# All 300 concepts x 5 trials at layer 15, coeff 6, temp 0.8 (computes any missing
# category vectors first) -> new_results/output_all_concepts_layer15_coeff6.csv:
modal run modal_app.py::all_concepts
modal volume get introspection-results new_results . --force
python judge_results.py --csv new_results/output_all_concepts_layer15_coeff6.csv --judges coherence affirmative_response affirmative_response_followed_by_correct_identification

# Per-category figures + summary tables (hit counts, cluster-bootstrap CIs):
python plot_by_category.py --csv new_results/output_all_concepts_layer15_coeff6.csv

# run_pca reads the GRADED csv from the results Volume (it selects the identified
# concepts), so push the judged copy back up before step 4:
modal volume put introspection-results new_results/output_all_concepts_layer15_coeff6.csv new_results/output_all_concepts_layer15_coeff6.csv --force
```

There is also a layer x coeff configuration sweep (20 concepts, layers 12/15/18, coeffs
4/6/9): `modal run modal_app.py::coherence_affirmation`, judged the same way with
`--judges coherence affirmative_response`.

## 4. PCA concept subspace

```bash
# Fit PCA over the normed vectors of concepts identified in >=1/5 trials of the
# all-concepts run, 80/20 train/test split (the projection sweeps run on the held-out
# test concepts) -> pca_subspace_all_concepts_layer15_coeff6.npz at the Volume root:
modal run modal_app.py::run_pca


# Local copies (used by plot_projection.py to recompute angles for older CSVs):
modal volume get introspection-results pca_subspace_all_concepts_layer15_coeff6.npz . --force

## 5. Projection alpha sweeps (held-out test concepts)

The projection CSV + judge_question txt land at the Volume root (`run_projection` runs
with cwd=/results), and `modal volume get` needs that remote path explicitly.

```bash
# Concept-vector sweep: rotate each held-out concept vector from inside the subspace
# (alpha=0) to orthogonal (alpha=1) -> projection_results_pca_subspace_..._coeff6.csv:
modal run modal_app.py::run_projection --ks "1 2 5 10 20" --interp-steps 7
modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6.csv ./new_results --force
modal volume get introspection-results judge_question_projection_results_pca_subspace_all_concepts_layer15_coeff6.txt ./new_results --force
python judge_results.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv --judges coherence affirmative_response affirmative_response_followed_by_correct_identification

# Random-direction control sweeps, seeds 0 and 1 (norm-matched random directions; their
# alpha=0.5 rows are the random-vector baseline). Identification is not judged -- a
# random direction has no ground-truth concept. Non-zero seeds get a _seed<N> suffix:
modal run modal_app.py::run_projection --random-vectors --ks "1 2 5 10 20" --interp-steps 7
modal run modal_app.py::run_projection --random-vectors --random-seed 1 --ks "1 2 5 10 20" --interp-steps 7
modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6_random.csv ./new_results --force
modal volume get introspection-results judge_question_projection_results_pca_subspace_all_concepts_layer15_coeff6_random.txt ./new_results --force
python judge_results.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random.csv --judges coherence affirmative_response
# (repeat the get + judge for the _random_seed1 files)

## 6. Random angle sweep (main figure)

Variants evenly spaced in angle to the subspace instead of mixing weight (an alpha
sweep of a random direction bunches everything at 70-90 degrees).

```bash
modal run modal_app.py::run_projection --sweep angle --random-vectors --interp-steps 7
modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random.csv ./new_results --force
modal volume get introspection-results judge_question_projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random.txt ./new_results --force
python judge_results.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random.csv --judges coherence affirmative_response

# Repeat for seed 1:
modal run modal_app.py::run_projection --sweep angle --random-vectors --random-seed 1 --interp-steps 7
modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.csv ./new_results --force
modal volume get introspection-results judge_question_projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.txt ./new_results --force
python judge_results.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.csv --judges coherence affirmative_response
```

## 7. Plotting & summary

Every projection plot carries the same four reference baselines so the figures are
comparable: no injection, all concepts, random vector, and "concepts identified >= 1/5
trials" -- the full-strength concept-vector rate over the sweep's test population, i.e.
only concepts the all-concepts run identified in at least 1 of their 5 trials.
Baselines are computed on the coherent & detected / coherent & correctly identified
judges (a trial only counts as detected/identified if it is ALSO coherent).

```bash
# Summary table of every baseline and experiment rate with sample counts:
python print_results_table.py

# MAIN two-panel figure (random angle sweep): k>=2 curves + mean-over-k with 95%
# cluster-bootstrap CIs (resampling base random directions), and the k=1 sweep split by
# the sign of each draw's PC1 component (reconstructed from random_seed + concept
# index). Defaults pool anglesweep seeds 0+1; the random-vector baseline is the
# alpha=0.5 rows of the alpha-sweep --random runs (= the unprojected direction, seeds
# 0+1 pooled), the concept-vector baseline the unfiltered 300-concept all-concepts run:
python plot_random_main_figure.py

# Angle sweep with random directions (seeds 0+1 pooled ->  _anglesweep_random_pooled2):
python plot_projection.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random.csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.csv --judges coherence+affirmative_response --no-injection-csv new_results/output_control_no_injection.csv --all-concepts-csv new_results/output_all_concepts_layer15_coeff6.csv --full-csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv --baseline-alpha 0.5

# Alpha-sweep random directions, seeds 0+1 pooled (-> _random_pooled2 plot)
# results at alpha=0.5 are used for as the (unprojected) random vector base line:
python plot_projection.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random.csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random_seed1.csv --judges coherence+affirmative_response --no-injection-csv new_results/output_control_no_injection.csv --all-concepts-csv new_results/output_all_concepts_layer15_coeff6.csv --full-csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv --baseline-alpha 0.5

# Concept-vector sweep (uncentered subspace; alpha=0.5 IS the full-vector direction,
# so --baseline-alpha 0.5 re-measures the full-vector baseline with k-times the trials):
python plot_projection.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv --judges coherence+affirmative_response coherence+affirmative_response_followed_by_correct_identification --no-injection-csv new_results/output_control_no_injection.csv --all-concepts-csv new_results/output_all_concepts_layer15_coeff6.csv --random-csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random.csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random_seed1.csv --baseline-alpha 0.5
```
