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
modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.csv ./new_results --force
# Get the relevant resulting files from modal to local
modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.csv ./new_results --force
modal volume get introspection-results judge_question_projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.txt ./new_results --force

# Judge results locally to not waste gpu time
python judge_results.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.csv --judges coherence affirmative_response

# Combine seed 0 and 1 in a single plot with control baselines:
python plot_projection.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random.csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6_anglesweep_random_seed1.csv --no-injection-csv new_results/output_control_no_injection.csv --all-concepts-csv new_results/output_all_concepts_layer15_coeff6.csv --full-csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv --baseline-alpha 0.5

