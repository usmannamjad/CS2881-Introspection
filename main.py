import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from inject_concept_vector import inject_concept_vector
import torch
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from api_utils import query_llm_judge
from all_prompts import get_anthropic_reproduce_messages, get_open_ended_belief_messages, get_generative_distinguish_messages, get_mcq_messages, get_injection_strength_messages
torch.manual_seed(2881)
# Distractors pool (randomly sampled words)
DISTRACTORS = ["Apple", "Zest", "Laughter", "Intelligence", "Vibrant", "Sad", "Beach", "Pottery", "Jewelry"]

ALL_JUDGES = ['coherence', 'thinking_about_word', 'affirmative_response', 'affirmative_response_followed_by_correct_identification']

def test_vector_multiple_choice(vector_path, model=None, tokenizer=None, max_new_tokens=100, type = 'anthropic_reproduce', coeff = 8.0, assistant_tokens_only = True, judges = None, temperature = 0.0, num_samples = 1):
    """
    Test a saved vector with a specific type of inference (to stress-test anthropic's introspection findings)
    Args:
        vector_path: Path to saved vector file from saved_vectors/llama/
        model: Loaded model (will load if None)
        tokenizer: Loaded tokenizer (will load if None)
        max_new_tokens: Max tokens for generation (100 if using original anthropic setup)
        type: 'anthropic_reproduce',  'mcq_knowledge' , 'mcq_distinguish','open_ended_belief', 'generative_distinguish', 'injection_strength'
        (types taken from anthropic SDF paper: https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/)
        judges: judge names to run per sample; pass an empty list to skip ALL API judge
        calls (including the mcq/injection_strength ones), e.g. to grade later with
        judge_results.py. None (default) runs all of ALL_JUDGES.
        num_samples: generations to produce in ONE batched generate() call (requires
        temperature > 0 when > 1). For mcq types the randomized options are shared
        across the batch.
    Returns:
        list of num_samples dicts, each with 'concept', 'layer', 'coeff', 'response', the judge responses
    """
    # Parse filename: concept_layer_avg.pt (concept may have underscores)
    filename = Path(vector_path).stem
    parts = filename.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")
    vec_type = parts[-1]  # 'avg' or 'last'
    layer = int(parts[-2])
    concept = '_'.join(parts[:-2])  # Handle concepts with underscores
    # Load vector and metadata
    if judges is None:
        judges = ALL_JUDGES
    data = torch.load(vector_path, weights_only=False)
    vector = data['vector']
    model_name = data.get('model_name', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
    
    # Load model if needed
    if model is None or tokenizer is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
    device = next(model.parameters()).device
    correct_letter = None
    options_text = None
    if type == 'mcq_knowledge' or type == 'mcq_distinguish':
        number_of_distractors = 9 if type == 'mcq_knowledge' else 1
        # Create multiple choice options
        distractors = random.sample([d for d in DISTRACTORS if d != concept], number_of_distractors)
        options = [concept] + distractors
        random.shuffle(options)
        correct_idx = options.index(concept)
        correct_letter = chr(65 + correct_idx)  # A-J 
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        messages = get_mcq_messages(options_text)
    elif type == "open_ended_belief":
        messages = get_open_ended_belief_messages()
    elif type == "generative_distinguish":
        messages = get_generative_distinguish_messages(concept)
    elif type == "anthropic_reproduce":
        messages = get_anthropic_reproduce_messages()
    elif type == "injection_strength":
        messages = get_injection_strength_messages()
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Find token position of double-newline before "Trial 1'. This ensures injection starts at "\n\nTrial 1" and continues through generation.
    # The prefilled "Ok." response (before this point) will NOT have injection.
    trial_start_text = "\n\nTrial 1"
    trial_start_pos = formatted_prompt.find(trial_start_text)
    if trial_start_pos != -1:
        # Tokenize up to that position to find the token index
        prefix = formatted_prompt[:trial_start_pos]
        injection_start_token = len(tokenizer.encode(prefix, add_special_tokens=False))
    else:
        # Fallback: use assistant_tokens_only behavior
        injection_start_token = None
    
    # Format full conversation for injection (assistant will generate after the last user message)
    # Perform inference with vector injection; num_samples > 1 batches all samples
    # into a single generate() call (same prompt + steering vector, sampling varies)
    responses = inject_concept_vector(
        model, tokenizer, vector, layer, coeff=coeff,
        inference_prompt=formatted_prompt, assistant_tokens_only=assistant_tokens_only,
        max_new_tokens=max_new_tokens, injection_start_token=injection_start_token,
        temperature=temperature, num_samples=num_samples
    )
    if isinstance(responses, str):
        responses = [responses]

    print(f"messages={messages}")

    # Format prompt for judge (without generation prompt)
    formatted_prompt_for_judge = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    question_text = "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?"

    # Expected strength category depends only on coeff, not on the response
    expected_strength_category = None
    if type == "injection_strength":
        if coeff < 5:
            expected_strength_category = "Weak"
        elif coeff < 10:
            expected_strength_category = "Moderate"
        elif coeff < 15:
            expected_strength_category = "Strong"
        else:
            expected_strength_category = "Very strong"

    results = []
    for response in responses:
        print(f"response={response}")

        # Run only the requested judges (others left as None). An empty judges list
        # skips every API call; the type-specific judges below are gated on it too.
        coherence_judge = query_llm_judge(question=formatted_prompt_for_judge, response=response, grading_type='coherence') if 'coherence' in judges else None
        thinking_about_word_judge = query_llm_judge(question=question_text, response=response, word=concept, grading_type='thinking_about_word') if 'thinking_about_word' in judges else None
        affirmative_response_judge = query_llm_judge(question=formatted_prompt_for_judge, response=response, grading_type='affirmative_response') if 'affirmative_response' in judges else None
        affirmative_response_followed_by_correct_identification_judge = query_llm_judge(question=question_text, response=response, word=concept, grading_type='affirmative_response_followed_by_correct_identification') if 'affirmative_response_followed_by_correct_identification' in judges else None

        # MCQ correctness judge (only for MCQ types)
        mcq_correct_judge = None
        if judges and type in ['mcq_knowledge', 'mcq_distinguish'] and correct_letter is not None and options_text is not None:
            mcq_correct_judge = query_llm_judge(response=response, grading_type='mcq_correct', options_text=options_text, correct_letter=correct_letter)

        # Judge if the model correctly identified the strength category
        injection_strength_correct_judge = None
        if judges and type == "injection_strength":
            injection_strength_correct_judge = query_llm_judge(response=response, grading_type='injection_strength_correct', expected_category=expected_strength_category)

        results.append({
            'concept': concept,
            'vec_type': vec_type,
            'layer': layer,
            'coeff': coeff,
            'type': type,
            'response': response,
            'coherence_judge': coherence_judge,
            'thinking_about_word_judge': thinking_about_word_judge,
            'affirmative_response_judge': affirmative_response_judge,
            'affirmative_response_followed_by_correct_identification_judge': affirmative_response_followed_by_correct_identification_judge,
            'mcq_correct_judge': mcq_correct_judge,
            'injection_strength_correct_judge': injection_strength_correct_judge,
            'expected_strength_category': expected_strength_category
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Run introspection experiments with concept vector injection")
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 18],
                       help="Layer indices to test (default: [15, 18])")
    parser.add_argument("--coeffs", type=float, nargs="+", default=[10, 12],
                       help="Coefficient values to test (default: [10, 12])")
    parser.add_argument("--type", type=str, default="anthropic_reproduce",
                       choices=["anthropic_reproduce", "mcq_knowledge", "mcq_distinguish", 
                               "open_ended_belief", "generative_distinguish", "injection_strength"],
                       help="Experiment type (default: injection_strength)")
    parser.add_argument("--assistant_tokens_only", action="store_true", default=True,
                       help="Only inject at assistant tokens (default: True)")
    parser.add_argument("--no_assistant_tokens_only", dest="assistant_tokens_only", action="store_false",
                       help="Inject at all tokens")
    parser.add_argument("--vectors_dir", type=str,
                       default="/n/home10/ehahami/work/nov26_experiments/saved_vectors/llama/",
                       help="Directory containing saved concept vector .pt files")
    parser.add_argument("--vec_types", type=str, nargs="+", default=["avg", "last"],
                       choices=["avg", "last"],
                       help="Which vector types to test (default: both avg and last)")
    parser.add_argument("--judges", type=str, nargs="+",
                       default=["coherence", "affirmative_response", "affirmative_response_followed_by_correct_identification"],
                       choices=ALL_JUDGES + ["none"],
                       help="Which of the 4 general judges to run (default: coherence, affirmative_response, "
                            "affirmative_response_followed_by_correct_identification; thinking_about_word excluded). "
                            "Pass 'none' to skip all judge API calls (grade afterwards with judge_results.py)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                       help="Skip (concept, vec_type, layer, coeff) combos already present in the results CSV (default: True)")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                       help="Rerun all combos even if already present in the results CSV")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Model name or path (loaded once and reused for all combos)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature for generation. 0.0 (default) => greedy/deterministic; "
                            ">0 => sample, so repeated trials vary (default: 0.0)")
    parser.add_argument("--trials_per_cell", type=int, default=1,
                       help="Number of generations per (concept, vec_type, layer, coeff) cell. Use >1 with "
                            "--temperature > 0 to gather a distribution of responses (default: 1)")
    parser.add_argument("--max_batch_size", type=int, default=25,
                       help="Max trials of one cell batched into a single generate() call when sampling "
                            "(temperature > 0). Cap keeps the KV cache well within A10G memory (default: 25)")
    parser.add_argument("--concepts", type=str, nargs="+", default=None,
                       help="Restrict the run to these concept names. If omitted, use every concept found "
                            "under --vectors_dir (default: None => all)")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Basename for output files (output_<run_name>.csv/.parquet, main_figure_<run_name>.png). "
                            "Defaults to the experiment type. Use a distinct name to avoid clobbering other runs "
                            "that share the same --type (e.g. a control run)")

    args = parser.parse_args()

    layers_to_test = args.layers
    coeffs_to_test = args.coeffs
    experiment_type = args.type
    assistant_tokens_only = args.assistant_tokens_only
    vec_types_to_test = args.vec_types
    # 'none' => empty list => no judge API calls at all (grade later with judge_results.py)
    judges_to_run = [] if "none" in args.judges else args.judges
    temperature = args.temperature
    trials_per_cell = args.trials_per_cell
    concepts_filter = set(args.concepts) if args.concepts else None
    run_label = args.run_name if args.run_name else experiment_type

    print(f"Testing layers: {layers_to_test}")
    print(f"Testing coefficients: {coeffs_to_test}")
    print(f"Experiment type: {experiment_type}")
    print(f"Assistant tokens only: {assistant_tokens_only}")
    print(f"Vector types: {vec_types_to_test}")
    print(f"Judges: {judges_to_run}")
    print(f"Temperature: {temperature}")
    print(f"Trials per cell: {trials_per_cell}")
    print(f"Concepts filter: {'all' if concepts_filter is None else sorted(concepts_filter)}")

    # Collect vectors by (concept, layer, vec_type)
    vectors_by_concept_layer = defaultdict(lambda: defaultdict(dict))
    for file in Path(args.vectors_dir).glob('*.pt'):
        filename = file.stem
        parts = filename.split('_')
        if len(parts) < 3:
            continue
        vec_type = parts[-1]  # 'avg' or 'last'
        layer = int(parts[-2])
        if layer in layers_to_test and vec_type in vec_types_to_test:
            concept = '_'.join(parts[:-2])
            if concepts_filter is not None and concept not in concepts_filter:
                continue
            vectors_by_concept_layer[concept][layer][vec_type] = file

    concepts = sorted(vectors_by_concept_layer.keys())
    print(f"Found {len(concepts)} concepts: {concepts}")
    # Print vec_types found for each concept
    for concept in concepts[:3]:  # Print first 3 as sample
        vec_types_found = set()
        for layer_dict in vectors_by_concept_layer[concept].values():
            vec_types_found.update(layer_dict.keys())
        print(f"  {concept}: vec_types = {sorted(vec_types_found)}")

    # Load model once (previously left as None, causing a full reload inside
    # test_vector_multiple_choice for every single combo)
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    # Store all results as a list of dictionaries (will convert to DataFrame)
    all_results = []

    # Set up incremental CSV saving
    results_dir = Path('new_results')
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f'output_{run_label}.csv'
    csv_initialized = False  # Track if CSV header has been written

    # The coherence/affirmative judges grade against the full chat-templated conversation.
    # For these experiment types it's identical for every row, so save it once next to the
    # CSV; judge_results.py reads this file instead of needing the (gated) tokenizer locally.
    judge_messages_by_type = {
        'anthropic_reproduce': get_anthropic_reproduce_messages,
        'open_ended_belief': get_open_ended_belief_messages,
        'injection_strength': get_injection_strength_messages,
    }
    if experiment_type in judge_messages_by_type:
        judge_question = tokenizer.apply_chat_template(
            judge_messages_by_type[experiment_type](), tokenize=False, add_generation_prompt=False)
        (results_dir / f'judge_question_{run_label}.txt').write_text(judge_question, encoding='utf-8')

    # Aggregate results per (layer, coeff, grader_type)
    # Structure: layer_results[layer][coeff][grader_type] = list of bools
    layer_results = defaultdict(lambda: defaultdict(lambda: {
        'coherence': [],
        'affirmative_response': [],
        'affirmative_response_followed_by_correct_identification': [],
        'thinking_about_word': [],
        'mcq_correct': [],
        'injection_strength_correct': []
    }))

    # If resuming, load already-run (concept, vec_type, layer, coeff) combos from the existing
    # CSV so we don't recompute them, and fold their judge results into the aggregates/output.
    existing_keys = set()
    if args.skip_existing and csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        for _, row in existing_df.iterrows():
            existing_keys.add((row['concept'], row['vec_type'], row['layer'], row['coeff'], int(row.get('trial', 0))))
            all_results.append(row.to_dict())
            for grader_type in ['coherence', 'affirmative_response', 'affirmative_response_followed_by_correct_identification',
                                 'thinking_about_word', 'mcq_correct', 'injection_strength_correct']:
                col = f'{grader_type}_judge'
                if col in row:
                    # Blank/NaN means that judge wasn't run for this row; keep it None
                    # rather than coercing to True (pd.isna(nan) is truthy under bool()).
                    value = row[col]
                    layer_results[row['layer']][row['coeff']][grader_type].append(
                        None if pd.isna(value) else bool(value))
        if existing_keys:
            print(f"Resuming: found {len(existing_keys)} already-run (concept, vec_type, layer, coeff) combos in {csv_path}")
            csv_initialized = True

    # Run experiments
    for concept in concepts:
        for layer in layers_to_test:
            if layer not in vectors_by_concept_layer[concept]:
                continue
            for vec_type in vectors_by_concept_layer[concept][layer]:
                vector_path = vectors_by_concept_layer[concept][layer][vec_type]

                for coeff in coeffs_to_test:
                    remaining_trials = [t for t in range(trials_per_cell)
                                        if (concept, vec_type, layer, coeff, t) not in existing_keys]
                    if len(remaining_trials) < trials_per_cell:
                        print(f"\nSkipping {trials_per_cell - len(remaining_trials)} already-run trial(s): "
                              f"{concept} at layer {layer} with vec_type {vec_type}, coeff {coeff}")

                    while remaining_trials:
                        # Sampling lets us batch several trials of the cell into one generate()
                        # call; greedy (temperature == 0) trials would be identical, so batch=1.
                        batch_size = args.max_batch_size if temperature > 0 else 1
                        batch = remaining_trials[:batch_size]
                        remaining_trials = remaining_trials[len(batch):]
                        print(f"\nTesting: {concept} at layer {layer} with vec_type {vec_type}, coeff {coeff}, "
                              f"trials {[t + 1 for t in batch]}/{trials_per_cell}")

                        batch_results = test_vector_multiple_choice(vector_path, model=model, tokenizer=tokenizer,
                                                                    coeff=coeff, type=experiment_type,
                                                                    assistant_tokens_only=assistant_tokens_only,
                                                                    judges=judges_to_run, temperature=temperature,
                                                                    num_samples=len(batch))

                        for trial, result in zip(batch, batch_results):
                            # Aggregate judge results by (layer, coeff, grader_type)
                            layer_results[layer][coeff]['coherence'].append(result['coherence_judge'])
                            layer_results[layer][coeff]['affirmative_response'].append(result['affirmative_response_judge'])
                            layer_results[layer][coeff]['affirmative_response_followed_by_correct_identification'].append(result['affirmative_response_followed_by_correct_identification_judge'])
                            layer_results[layer][coeff]['thinking_about_word'].append(result['thinking_about_word_judge'])
                            # Track MCQ correctness if available
                            if result.get('mcq_correct_judge') is not None:
                                layer_results[layer][coeff]['mcq_correct'].append(result['mcq_correct_judge'])
                            # Track injection strength correctness if available
                            if result.get('injection_strength_correct_judge') is not None:
                                layer_results[layer][coeff]['injection_strength_correct'].append(result['injection_strength_correct_judge'])

                            # Store result for DataFrame. Judges that weren't run (not requested via
                            # --judges, or not applicable to this experiment type) are kept as None
                            # so the CSV distinguishes "not judged" from an actual False verdict.
                            result_row = {
                                'concept': result['concept'],
                                'vec_type': result.get('vec_type', ''),
                                'layer': result['layer'],
                                'coeff': result['coeff'],
                                'trial': trial,
                                'temperature': temperature,
                                'type': result.get('type', ''),
                                'assistant_tokens_only': assistant_tokens_only,
                                'coherence_judge': result['coherence_judge'],
                                'thinking_about_word_judge': result['thinking_about_word_judge'],
                                'affirmative_response_judge': result['affirmative_response_judge'],
                                'affirmative_response_followed_by_correct_identification_judge': result['affirmative_response_followed_by_correct_identification_judge'],
                                'mcq_correct_judge': result.get('mcq_correct_judge'),
                                'injection_strength_correct_judge': result.get('injection_strength_correct_judge'),
                                'expected_strength_category': result.get('expected_strength_category', ''),
                                'response': result['response']
                            }
                            all_results.append(result_row)

                            # Save incrementally to CSV
                            result_df = pd.DataFrame([result_row])
                            if not csv_initialized:
                                # Write with header (first time)
                                result_df.to_csv(csv_path, index=False, mode='w')
                                csv_initialized = True
                            else:
                                # Append without header
                                result_df.to_csv(csv_path, index=False, mode='a', header=False)

    # Save final results as DataFrame (CSV already saved incrementally, but save full version for Parquet)
    results_df = pd.DataFrame(all_results)
    
    # CSV already saved incrementally, but save full version to ensure consistency
    results_df.to_csv(csv_path, index=False)
    print(f"\nFinal results saved to {csv_path}")
    
    # Save as Parquet (more efficient, preserves types, better for large datasets)
    try:
        parquet_path = results_dir / f'output_{run_label}.parquet'
        results_df.to_parquet(parquet_path, index=False)
        print(f"Results saved to {parquet_path}")
    except ImportError:
        print("Note: pyarrow not installed, skipping Parquet export. Install with: pip install pyarrow")

    # Compute rates per (layer, coeff, grader_type), aggregating over all concepts x trials
    # in the cell. Also track the sample count n so we can draw binomial standard-error bars.
    # Structure: rates[layer][coeff][grader_type] = rate ; counts[...] = n
    rates = defaultdict(lambda: defaultdict(dict))
    counts = defaultdict(lambda: defaultdict(dict))

    grader_types = ['coherence', 'affirmative_response', 'affirmative_response_followed_by_correct_identification', 'thinking_about_word', 'mcq_correct', 'injection_strength_correct']

    for layer in layers_to_test:
        for coeff in coeffs_to_test:
            metrics = layer_results[layer][coeff]
            for grader_type in grader_types:
                # Judges that weren't requested (not in --judges) still get a None
                # appended as a placeholder; exclude those so counts reflect only
                # judges that actually ran.
                values = [v for v in metrics[grader_type] if v is not None]
                n = len(values)
                rate = sum(values) / n if n else 0.0
                rates[layer][coeff][grader_type] = rate
                counts[layer][coeff][grader_type] = n
        print(f"Layer {layer} rates computed")

    # Only plot grader types that actually produced samples (e.g. coherence &
    # affirmative_response for these runs); this drops the flat-zero clutter from
    # graders that weren't run (mcq, injection_strength, thinking_about_word, ...).
    active_graders = [g for g in grader_types
                      if any(counts[l][c][g] > 0 for l in layers_to_test for c in coeffs_to_test)]
    if not active_graders:
        print("No judge results to plot (judges skipped); grade and plot locally with judge_results.py")
        return
    # Representative per-cell sample count (concepts x trials in one layer/coeff cell).
    per_cell_n = max((counts[l][c][active_graders[0]] for l in layers_to_test for c in coeffs_to_test),
                     default=0) if active_graders else 0

    # Plot results: separate line for each (coeff, grader_type) combination.
    # Rate is aggregated over concepts x trials per cell; error bars are the binomial
    # standard error sqrt(p*(1-p)/n) over those aggregated samples.
    layers = sorted(layers_to_test)
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    plt.figure(figsize=(14, 8))

    # Plot each combination that actually has samples (skip coeff/grader pairs
    # with zero real judge results across all layers, e.g. a coeff only run
    # under a different grader set).
    for coeff in coeffs_to_test:
        for idx, grader_type in enumerate(active_graders):
            if not any(counts[l][coeff][grader_type] > 0 for l in layers):
                continue
            y_values = [rates[l][coeff][grader_type] for l in layers]
            y_err = [
                (rates[l][coeff][grader_type] * (1 - rates[l][coeff][grader_type]) / counts[l][coeff][grader_type]) ** 0.5
                if counts[l][coeff][grader_type] > 0 else 0.0
                for l in layers
            ]
            label = f'coeff={coeff}, {grader_type}'
            plt.errorbar(layers, y_values, yerr=y_err, marker=markers[idx % len(markers)],
                    linestyle=linestyles[coeffs_to_test.index(coeff) % len(linestyles)],
                    label=label, linewidth=2, markersize=6, capsize=3)

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Rate (mean over concepts x trials)', fontsize=12)
    plt.title(f'Experiment: {run_label} (type={experiment_type}, temp={temperature}, '
              f'trials/cell={trials_per_cell}, n={per_cell_n}/cell)', fontsize=13)
    # Single layer (e.g. control) has no x-range to sweep; make the lone point readable.
    if len(layers) == 1:
        plt.xlim(layers[0] - 1, layers[0] + 1)
        plt.xticks(layers)
    plt.legend(fontsize=8, ncol=2, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save figure to plots folder
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    figure_path = plots_dir / f'main_figure_{run_label}.png'
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {figure_path}")
    plt.close()  # Close instead of show for batch jobs

if __name__ == "__main__":
    main()
    
