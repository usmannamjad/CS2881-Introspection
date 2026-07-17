# 1. Compute simple_data vectors (won't recompute if .pt files already exist)
python save_vectors.py --save_dir ./saved_vectors/llama

# 2. Run the anthropic_reproduce trial: layer 15, coeff 9, avg vectors only
python main.py --type anthropic_reproduce --layers 15 --coeffs 9 \
  --vectors_dir ./saved_vectors/llama --vec_types avg


Simple data uses 50 concept words + 250 category specific concept words from Jack Lindsey's "Emergent Introspective Awareness in Large Language Models" 2025 paper. Overlapping or near overlapping concepts where replaced with new ones.

old_results and old_plots contains previous results by Hahami et al., new_results and plots has our results.

Added the following to the coherence judge prompt since a significant number of coherent responses were judged incoherent in the control experiment: "I don't detect an injected thought on Trial 1. I'll wait for the next trial to see if I can pick up on the injected pattern.". Also added four examples to the prompt.

To reproduce main projection results:
# Run angle sweep on modal:
modal run modal_app.py::run_projection --sweep angle --random-vectors
# Get the relevant resulting files from modal to local
modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random.csv ./new_results --force
modal volume get introspection-results judge_question_projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random.txt ./new_results --force
# Judge results locally to not waste gpu time
python judge_results.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random.csv --judges coherence affirmative_response


# Repeat for seed 1
# Run angle sweep on modal:
modal run modal_app.py::run_projection --sweep angle --random-vectors --random-seed 1
# Get the relevant resulting files from modal to local
modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.csv ./new_results --force
modal volume get introspection-results judge_question_projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.txt ./new_results --force

# Judge results locally to not waste gpu time
python judge_results.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.csv --judges coherence affirmative_response

# Plotting. Every projection plot carries the same four reference baselines so the
# figures are comparable: no injection, all concepts, random vector, and
# "concepts identified >= 1/5 trials" -- the full-strength concept-vector rate over the
# sweep's test population, i.e. only concepts the all-concepts run identified in at
# least 1 of their 5 trials. Baselines are computed on the coherence-ANDed judges
# (coherent & detected, coherent & correctly identified) -- the same metric as the
# panel they are drawn on (see the y label), never raw detection -- and legends carry
# only the rate; sample sizes are printed to the console for captions. Each command
# also writes an _angle companion figure.

# Adding --paper to any plot_projection.py command below writes an ICML-style variant
# (..._paper.png): baselines direct-labeled at the clearer end of their line, no rate
# numbers, no baseline legend box.

# MAIN two-panel figure (random angle sweep): k>=2 curves + mean-over-k with 95%
# cluster-bootstrap CIs (resampling base random directions), and the k=1 sweep split by
# the sign of each draw's PC1 component (reconstructed from random_seed + concept
# index). Pass both anglesweep CSVs to --csv once seed 1 is judged:
python plot_random_main_figure.py

# Angle sweep with random directions (seed 0; add the seed 1 CSV to --csv once it is
# judged to pool them into an _pooled2 plot):
python plot_projection.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random.csv --judges coherence+affirmative_response --no-injection-csv new_results/output_control_no_injection.csv --all-concepts-csv new_results/output_all_concepts_layer15_coeff6.csv --full-csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv --baseline-alpha 0.5

# Alpha-sweep random directions, seeds 0+1 pooled (-> _random_pooled2 plot):
python plot_projection.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random.csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random_seed1.csv --judges coherence+affirmative_response --no-injection-csv new_results/output_control_no_injection.csv --all-concepts-csv new_results/output_all_concepts_layer15_coeff6.csv --full-csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv --baseline-alpha 0.5

# Concept-vector sweep (uncentered subspace; alpha=0.5 IS the full-vector direction,
# so --baseline-alpha 0.5 re-measures the full-vector baseline with k-times the trials):
python plot_projection.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv --judges coherence+affirmative_response coherence+affirmative_response_followed_by_correct_identification --no-injection-csv new_results/output_control_no_injection.csv --all-concepts-csv new_results/output_all_concepts_layer15_coeff6.csv --random-csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random.csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random_seed1.csv --baseline-alpha 0.5

# Concept-vector sweep, centered subspace (no --baseline-alpha: for a centered subspace
# alpha=0.5 is NOT the full-vector direction, so baselines use the true 'full' rows):
python plot_projection.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_centered.csv --judges coherence+affirmative_response coherence+affirmative_response_followed_by_correct_identification --no-injection-csv new_results/output_control_no_injection.csv --all-concepts-csv new_results/output_all_concepts_layer15_coeff6.csv --random-csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random.csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_random_seed1.csv

