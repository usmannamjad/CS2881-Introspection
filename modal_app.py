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

Download results afterwards:
    modal volume get introspection-results new_results ./new_results_modal
    modal volume get introspection-results plots ./plots_modal
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
):
    # cwd=/results so main.py's relative new_results/ and plots/ land on the Volume
    subprocess.run(
        [
            "python", "/root/app/main.py",
            "--type", experiment_type,
            "--layers", *layers.split(),
            "--coeffs", *coeffs.split(),
            "--vectors_dir", "/vectors/llama",
            "--vec_types", *vec_types.split(),
        ],
        check=True,
        cwd="/results",
    )
    results_vol.commit()


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
