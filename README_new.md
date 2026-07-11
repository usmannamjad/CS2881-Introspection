# 1. Compute simple_data vectors (won't recompute if .pt files already exist)
python save_vectors.py --save_dir ./saved_vectors/llama

# 2. Run the anthropic_reproduce trial: layer 15, coeff 9, avg vectors only
python main.py --type anthropic_reproduce --layers 15 --coeffs 9 \
  --vectors_dir ./saved_vectors/llama --vec_types avg


Simple data uses 50 concept words + 250 category specific concept words from Jack Lindsey's "Emergent Introspective Awareness in Large Language Models" 2025 paper. Overlapping or near overlapping concepts where replaced with new ones.

old_results contains previous results by Hahami et al., new_results has our results.