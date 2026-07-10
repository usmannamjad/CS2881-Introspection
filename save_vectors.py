from compute_concept_vector_utils import compute_concept_vectors_all_layers
from inject_concept_vector import inject_concept_vector
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import numpy as np
from pathlib import Path
def sweep_all_layers_and_coefficients(model, tokenizer, model_name, datasets, layer_range, save_dir):
    """Sweep all layers to compute concept vectors for all concepts in all datasets"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        # print(f"\n{'='*80}")
        # print(f"Processing dataset: {dataset_name}")
        # print(f"{'='*80}")
        # Compute concept vectors for all layers in one sweep over the data
        # (one forward pass per prompt instead of one per layer)
        steering_vectors_by_layer = compute_concept_vectors_all_layers(model, tokenizer, dataset_name, layer_range)
        for layer_idx, steering_vectors in steering_vectors_by_layer.items():
            for concept_name, (vec_last, vec_avg) in steering_vectors.items():
                # Process both vec_last and vec_avg
                for vec_type, steering_vector in [("last", vec_last), ("avg", vec_avg)]:
                        # Save all vectors with metadata
                        filename = f"{concept_name}_{layer_idx}_{vec_type}.pt"
                        filepath = save_path / filename
                        save_data = {
                            'vector': steering_vector,
                            'model_name': model_name,
                            'concept_name': concept_name,
                            'layer': layer_idx,
                            'vec_type': vec_type
                        }
                        torch.save(save_data, filepath)
def main():
    parser = argparse.ArgumentParser(description="Sweep layers and coefficients for concept vector injection")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Model name or path")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       default=["simple_data", "complex_data"],
                       help="Datasets to process")
    parser.add_argument("--layer_range", type=int, nargs="+",
                       default=list(range(32)),
                       help="Layer indices to sweep (default: 0-31)")
    parser.add_argument("--save_dir", type=str,
                       default="/n/home10/ehahami/work/nov26_experiments/saved_vectors/llama",
                       help="Directory to save vectors")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")
    
    sweep_all_layers_and_coefficients(model, tokenizer, args.model, args.datasets, args.layer_range, args.save_dir)

if __name__ == "__main__":
    main()
