"""Modal wrapper for the introspection experiments.

Setup (once):
    pip install modal
    modal setup
    modal secret create openai-secret OPENAI_API_KEY=sk-...
    modal secret create huggingface-secret HF_TOKEN=hf_...

Run:
    modal run modal_app.py                      # vectors (all layers) + experiment (layer 15, coeff 9, avg)
    modal run modal_app.py --step vectors       # only compute/save vectors
    modal run modal_app.py --step experiment    # only run the injection trial
    modal run modal_app.py --layers "12 15 18" --coeffs "4 9 16" --vec-types "avg last"
    # --layers controls the experiment; --vector-layers (default "all") controls save_vectors

    # Coherence & affirmation experiments (require vectors to exist first; run --step vectors if not):
    modal run modal_app.py::control                 # control: no injection, 100 samples, temp 0.8
    modal run modal_app.py::coherence_affirmation   # 20 concepts x layers[12,15,18] x alpha[4,6,9], temp 0.8, 5 trials/cell

Download results afterwards (target the current dir '.', not './new_results':
passing an existing dir makes modal nest the download as ./new_results/new_results.
Add --force to overwrite on a re-download):
    modal volume get introspection-results new_results .
    modal volume get introspection-results plots .
"""
import subprocess
import modal

app = modal.App("cs2881-introspection")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "openai",
        "pandas",
        "matplotlib",
        "tqdm",
        "pyarrow",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    # Repo code baked into the image; results/vectors live on Volumes instead
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=[".git/**", "__pycache__/**", "new_results/**", "plots/**", "*.pt"],
    )
)

# Persist HF model weights, computed vectors, and experiment outputs across runs
hf_cache_vol = modal.Volume.from_name("introspection-hf-cache", create_if_missing=True)
vectors_vol = modal.Volume.from_name("introspection-vectors", create_if_missing=True)
results_vol = modal.Volume.from_name("introspection-results", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/vectors": vectors_vol,
    "/results": results_vol,
}

# main.py / save_vectors.py load Llama-3.1-8B in float16 (~16 GB), which fits an
# A10G (24 GB) comfortably. Bump to L40S/A100 only if you batch or switch to fp32.
GPU = "A10G"


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 3600,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def save_vectors(layers: str = "all", datasets: str = "simple_data"):
    # "all" = omit --layer_range so save_vectors.py's default (layers 0-31) applies.
    # All layers cost the same as one: activations for every layer come from the
    # same single forward pass per prompt.
    layer_args = [] if layers == "all" else ["--layer_range", *layers.split()]
    subprocess.run(
        [
            "python", "/root/app/save_vectors.py",
            *layer_args,
            "--datasets", *datasets.split(),
            "--save_dir", "/vectors/llama",
        ],
        check=True,
    )
    vectors_vol.commit()


@app.function(
    image=image,
    gpu=GPU,
    timeout=4 * 3600,
    volumes=VOLUMES,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("openai-secret"),
    ],
)
def run_experiment(
    layers: str = "15",
    coeffs: str = "9",
    experiment_type: str = "anthropic_reproduce",
    vec_types: str = "avg",
    skip_existing: bool = False,
    temperature: float = 0.0,
    trials_per_cell: int = 1,
    concepts: str = "",
    judges: str = "",
    run_name: str = "",
):
    cmd = [
        "python", "/root/app/main.py",
        "--type", experiment_type,
        "--layers", *layers.split(),
        "--coeffs", *coeffs.split(),
        "--vectors_dir", "/vectors/llama",
        "--vec_types", *vec_types.split(),
        "--temperature", str(temperature),
        "--trials_per_cell", str(trials_per_cell),
        "--skip_existing" if skip_existing else "--no_skip_existing",
    ]
    if concepts:
        cmd += ["--concepts", *concepts.split()]
    if judges:
        cmd += ["--judges", *judges.split()]
    if run_name:
        cmd += ["--run_name", run_name]
    # cwd=/results so main.py's relative new_results/ and plots/ land on the Volume
    subprocess.run(cmd, check=True, cwd="/results")
    results_vol.commit()


# 20 concepts for the coherence/affirmation sweep. The first 5 are concepts that
# produced affirmative "injected thought" responses in the prior anthropic_reproduce
# results (new_results, now archived under old_results); the remaining 15 round the
# set out to 20. All are drawn from dataset/simple_data.json's concept_vector_words,
# so save_vectors produces a vector for each.
COHERENCE_AFFIRMATION_CONCEPTS = [
    # 5 that were affirmative previously
    "Origami", "Satellites", "Dust", "Illusions", "Trumpets",
    # 15 more to reach 20
    "Cameras", "Lightning", "Constellations", "Treasures", "Phones",
    "Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
    "Sadness", "Xylophones", "Secrecy", "Oceans", "Information",
]


@app.local_entrypoint()
def control(
    samples: int = 100,
    temperature: float = 0.8,
    concept: str = "Dust",
    layer: str = "15",
):
    """Control: coherence & affirmation with NO concept injected.

    Runs `samples` generations of the anthropic_reproduce prompt at coeff=0 (the
    steering vector is added with zero weight, i.e. no injection) and temperature
    `temperature`, judging coherence and affirmation. This measures the baseline /
    false-positive rate when nothing is injected.

        modal run modal_app.py::control                 # 100 samples, temp 0.8
        modal run modal_app.py::control --samples 200

    Results: new_results/output_control_no_injection.csv on the results Volume.
    """
    print(f"Running control (no injection): {samples} samples, temperature={temperature}")
    run_experiment.remote(
        layers=layer,
        coeffs="0",
        experiment_type="anthropic_reproduce",
        vec_types="avg",
        temperature=temperature,
        trials_per_cell=samples,
        concepts=concept,  # coeff=0 => which concept is loaded doesn't affect output
        judges="coherence affirmative_response",
        run_name="control_no_injection",
    )


@app.local_entrypoint()
def coherence_affirmation(
    temperature: float = 0.8,
    trials_per_cell: int = 5,
):
    """Coherence & affirmation sweep over 20 concepts x layers x alphas.

    vector_type=avg, concepts=20 (5 previously-affirmative + 15 more),
    layers=[12,15,18], alpha=[4,6,9], temperature=0.8, trials_per_cell=5.

        modal run modal_app.py::coherence_affirmation

    Results: new_results/output_coherence_affirmation.csv on the results Volume.
    """
    concepts = " ".join(COHERENCE_AFFIRMATION_CONCEPTS)
    print(
        f"Running coherence/affirmation sweep: {len(COHERENCE_AFFIRMATION_CONCEPTS)} concepts, "
        f"layers=[12,15,18], alpha=[4,6,9], temperature={temperature}, trials_per_cell={trials_per_cell}"
    )
    run_experiment.remote(
        layers="12 15 18",
        coeffs="4 6 9",
        experiment_type="anthropic_reproduce",
        vec_types="avg",
        temperature=temperature,
        trials_per_cell=trials_per_cell,
        concepts=concepts,
        judges="coherence affirmative_response",
        run_name="coherence_affirmation",
    )


@app.local_entrypoint()
def main(
    step: str = "all",
    layers: str = "15",
    vector_layers: str = "all",
    coeffs: str = "9",
    experiment_type: str = "anthropic_reproduce",
    vec_types: str = "avg",
    datasets: str = "simple_data",
):
    if step not in ("all", "vectors", "experiment"):
        raise ValueError(f"Unknown step: {step} (use 'all', 'vectors', or 'experiment')")
    if step in ("all", "vectors"):
        print(f"Computing vectors: layers={vector_layers}, datasets={datasets}")
        save_vectors.remote(layers=vector_layers, datasets=datasets)
    if step in ("all", "experiment"):
        print(f"Running experiment: type={experiment_type}, layers={layers}, coeffs={coeffs}, vec_types={vec_types}")
        run_experiment.remote(
            layers=layers,
            coeffs=coeffs,
            experiment_type=experiment_type,
            vec_types=vec_types,
        )
