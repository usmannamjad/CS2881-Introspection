from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import argparse
import numpy as np
from pathlib import Path

def get_model_type(tokenizer):
    """Detect model type from tokenizer (llama or qwen)"""
    model_name = tokenizer.name_or_path.lower()
    if "qwen" in model_name:
        return "qwen"
    else:
        return "llama"

def format_prompt(model_type, user_message, dataset_name=None):
    """Format prompt based on model type"""
    if model_type == "qwen":
        if dataset_name == "simple_data":
            return f"<|im_start|>user\nTell me about {user_message}.<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    else:  # llama
        if dataset_name == "simple_data":
            return f"<|start_header_id|>user<|end_header_id|>Tell me about {user_message}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            return f"<|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def get_data(dataset_name): 
    """Load raw data from json files"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dataset"
    
    if dataset_name == "simple_data":
        with open(dataset_dir / "simple_data.json", "r") as f:
            data = json.load(f)
        return data
    elif dataset_name == "complex_data":
        with open(dataset_dir / "complex_data.json", "r") as f:
            data = json.load(f)
        return data


def compute_vectors_single_prompt(model, tokenizer, dataset_name, steering_prompt, layer_indices):
    """
    Compute activation vectors for a single prompt/sentence across multiple layers,
    using a single forward pass (output_hidden_states=True already returns every
    layer's hidden states, so all layer_indices can be read from one pass).

    Returns:
        dict: {layer_idx: (prompt_last_vector, prompt_average_vector)}
    """
    model_type = get_model_type(tokenizer)
    prompt = format_prompt(model_type, steering_prompt, dataset_name)

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) # [batch_size, seq_len, hidden_dim]
        vectors = {}
        for layer_idx in layer_indices:
            hidden = outputs.hidden_states[layer_idx]
            # like the ":" token in "Assistant:", but for llama it is <end_header_id> token
            prompt_last_vector = hidden[:, prompt_len - 1, :].detach().cpu()
            prompt_average_vector = hidden[:, :prompt_len, :].mean(dim=1).detach().cpu()
            vectors[layer_idx] = (prompt_last_vector, prompt_average_vector)
        del outputs

    return vectors

def compute_vector_single_prompt(model, tokenizer, dataset_name, steering_prompt, layer_idx):
    """
    Compute activation vector for a single prompt/sentence at a single layer
    Based on Anthropic's introspection paper methodology

    Returns:
        prompt_last_vector: activation at last token (e.g., <end_header_id>)
        prompt_average_vector: average activation across all prompt tokens
    """
    return compute_vectors_single_prompt(model, tokenizer, dataset_name, steering_prompt, [layer_idx])[layer_idx]

def compute_concept_vectors_all_layers(model, tokenizer, dataset_name, layer_indices, save_dir=None, skip_existing=True):
    """
    Compute steering vectors for all concepts in the dataset, for every layer in
    layer_indices, doing only one forward pass per prompt (not one per layer).

    Args:
        model: the model to use
        tokenizer: the tokenizer to use
        dataset_name: "simple_data" or "complex_data"
        layer_indices: the layer indices to compute steering vectors for
        save_dir: directory where vectors are (or will be) saved as {concept}_{layer}_{vec_type}.pt.
            Used to detect already-computed vectors when skip_existing is True.
        skip_existing: if True (default) and save_dir is given, concepts whose vector files
            already exist on disk for every requested layer/vec_type are not recomputed.

    Returns:
        dict: {layer_idx: {concept_name: [prompt_last_steering_vector, prompt_average_steering_vector]}}
        Concepts skipped because they already exist on disk are omitted.

    Method:
        - simple_data: For each word: vector(word) - mean(vector(baseline_word) for all baselines)
        - complex_data: For each concept: mean(vectors(pos_sentences)) - mean(vectors(neg_sentences))
    """
    data = get_data(dataset_name)
    steering_vectors = {layer_idx: {} for layer_idx in layer_indices}

    def already_computed(name):
        if not skip_existing or save_dir is None:
            return False
        save_path = Path(save_dir)
        return all((save_path / f"{name}_{layer_idx}_{vec_type}.pt").exists()
                   for layer_idx in layer_indices for vec_type in ("last", "avg"))

    if dataset_name == "simple_data":
        all_concept_words = data["concept_vector_words"]
        concept_words = [w for w in all_concept_words if not already_computed(w)]
        skipped = set(all_concept_words) - set(concept_words)
        if skipped:
            print(f"Skipping {len(skipped)} already-computed concept(s): {sorted(skipped)}")
        if not concept_words:
            return steering_vectors
        baseline_words = data["baseline_words"][:50]

        # Compute baseline means once (used for all concepts, all layers)
        print(f"Computing baseline mean from {len(baseline_words)} words...")
        baseline_vecs_last = {layer_idx: [] for layer_idx in layer_indices}
        baseline_vecs_avg = {layer_idx: [] for layer_idx in layer_indices}
        for word in tqdm(baseline_words, desc="Baseline vectors"):
            vectors = compute_vectors_single_prompt(model, tokenizer, dataset_name, word, layer_indices)
            for layer_idx in layer_indices:
                vec_last, vec_avg = vectors[layer_idx]
                baseline_vecs_last[layer_idx].append(vec_last)
                baseline_vecs_avg[layer_idx].append(vec_avg)
        baseline_mean_last = {layer_idx: torch.stack(baseline_vecs_last[layer_idx], dim=0).mean(dim=0).squeeze()
                               for layer_idx in layer_indices} # shape [hidden_dim]
        baseline_mean_avg = {layer_idx: torch.stack(baseline_vecs_avg[layer_idx], dim=0).mean(dim=0).squeeze()
                              for layer_idx in layer_indices} # shape [hidden_dim]

        # Compute steering vectors for each concept word
        for word in tqdm(concept_words, desc="Concept vectors"):
            vectors = compute_vectors_single_prompt(model, tokenizer, dataset_name, word, layer_indices)
            for layer_idx in layer_indices:
                vec_last, vec_avg = vectors[layer_idx]
                vec_last = vec_last.squeeze() # shape [hidden_dim]
                vec_avg = vec_avg.squeeze() # shape [hidden_dim]
                steering_vectors[layer_idx][word] = [vec_last - baseline_mean_last[layer_idx],
                                                      vec_avg - baseline_mean_avg[layer_idx]]

    elif dataset_name == "complex_data":
        # For each concept: mean(positive) - mean(negative)
        all_concepts = list(data.keys())
        concepts = [c for c in all_concepts if not already_computed(c)]
        skipped = set(all_concepts) - set(concepts)
        if skipped:
            print(f"Skipping {len(skipped)} already-computed concept(s): {sorted(skipped)}")
        print(f"data keys: {data.keys()}")
        for concept_name in concepts:
            print(f"concept_name: {concept_name}")
            pos_sentences = data[concept_name][0]  # List of positive examples
            neg_sentences = data[concept_name][1]  # List of negative examples

            print(f"\nProcessing {concept_name}: {len(pos_sentences)} pos, {len(neg_sentences)} neg")

            # Compute mean of positive sentences
            pos_vecs_last = {layer_idx: [] for layer_idx in layer_indices}
            pos_vecs_avg = {layer_idx: [] for layer_idx in layer_indices}
            for sentence in tqdm(pos_sentences, desc=f"{concept_name} (positive)"):
                vectors = compute_vectors_single_prompt(model, tokenizer, dataset_name, sentence, layer_indices)
                for layer_idx in layer_indices:
                    vec_last, vec_avg = vectors[layer_idx]
                    pos_vecs_last[layer_idx].append(vec_last)
                    pos_vecs_avg[layer_idx].append(vec_avg)
            pos_mean_last = {layer_idx: torch.stack(pos_vecs_last[layer_idx], dim=0).mean(dim=0).squeeze()
                             for layer_idx in layer_indices}
            pos_mean_avg = {layer_idx: torch.stack(pos_vecs_avg[layer_idx], dim=0).mean(dim=0).squeeze()
                            for layer_idx in layer_indices}

            # Compute mean of negative sentences
            neg_vecs_last = {layer_idx: [] for layer_idx in layer_indices}
            neg_vecs_avg = {layer_idx: [] for layer_idx in layer_indices}
            for sentence in tqdm(neg_sentences, desc=f"{concept_name} (negative)"):
                vectors = compute_vectors_single_prompt(model, tokenizer, dataset_name, sentence, layer_indices)
                for layer_idx in layer_indices:
                    vec_last, vec_avg = vectors[layer_idx]
                    neg_vecs_last[layer_idx].append(vec_last)
                    neg_vecs_avg[layer_idx].append(vec_avg)
            neg_mean_last = {layer_idx: torch.stack(neg_vecs_last[layer_idx], dim=0).mean(dim=0).squeeze()
                             for layer_idx in layer_indices}
            neg_mean_avg = {layer_idx: torch.stack(neg_vecs_avg[layer_idx], dim=0).mean(dim=0).squeeze()
                            for layer_idx in layer_indices}

            # Steering vectors = positive - negative (both last and avg)
            for layer_idx in layer_indices:
                steering_vectors[layer_idx][concept_name] = [pos_mean_last[layer_idx] - neg_mean_last[layer_idx],
                                                               pos_mean_avg[layer_idx] - neg_mean_avg[layer_idx]]

    for layer_idx in layer_indices:
        print(f"\nLayer {layer_idx}: computed {len(steering_vectors[layer_idx])} steering vectors (each with last and avg variants)")
    return steering_vectors

def compute_concept_vector(model, tokenizer, dataset_name, layer_idx):
    """
    Compute steering vectors for all concepts in the dataset, for a single layer.

    Args:
        model: the model to use
        tokenizer: the tokenizer to use
        dataset_name: "simple_data" or "complex_data"
        layer_idx: the layer index to compute steering vectors for

    Returns:
        dict: {concept_name: [prompt_last_steering_vector, prompt_average_steering_vector]}
    """
    return compute_concept_vectors_all_layers(model, tokenizer, dataset_name, [layer_idx])[layer_idx]