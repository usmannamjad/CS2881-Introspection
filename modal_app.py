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

    The GPU runs generate responses only (judges='none'): sampled trials of a cell are
    batched into a single generate() call, and the OpenAI judge calls -- the slow,
    GPU-idle part -- happen locally afterwards via judge_results.py.

    Download results afterwards (target the current dir '.', not './new_results':
    passing an existing dir makes modal nest the download as ./new_results/new_results.
    Add --force to overwrite on a re-download):
        modal volume get introspection-results new_results .
        modal volume get introspection-results plots .

    Then grade + plot locally (needs OPENAI_API_KEY; resumable, only fills empty judge columns):
    python judge_results.py --csv new_results/output_coherence_affirmation.csv
   
    
    # All concept vectors and PCA experiments commands:
    modal run modal_app.py::all_concepts            # all 300 concepts (50 main + 5 categories x 50) x layer 15 x alpha 6, avg
                                                    # (computes the missing category vectors first; judge with
                                                    #  judge_results.py --judges coherence affirmative_response_followed_by_correct_identification)
    # Judge all_concept results locally
    modal volume get introspection-results new_results . --force
    python judge_results.py --csv new_results/output_all_concepts_layer15_coeff6.csv --judges coherence affirmative_response_followed_by_correct_identification
    # Put locally judged results on modal
    modal volume put introspection-results new_results/output_all_concepts_layer15_coeff6.csv new_results/output_all_concepts_layer15_coeff6.csv --force
    # PCA subspace of identified-concept vectors + projection experiment (run after
    # all_concepts is judged; run_pca reads new_results/output_*.csv from the Volume):
    modal run modal_app.py::run_pca                 # fit subspace on TRAIN identified concepts, hold out TEST
                                                    #   -> /results/pca_subspace_all_concepts_layer15_coeff6.npz
    modal run modal_app.py::run_projection          # inject proj->orthogonal interpolation on held-out concepts
                                                    #   (generate only; grade + plot locally afterwards like all_concepts)
    modal run modal_app.py::run_projection --random-vectors
                                                    # control: random directions norm-matched to the held-out concept
                                                    #   vectors -- base random detection rate ('full') plus the same
                                                    #   proj->orthogonal sweep of each random direction
                                                    #   -> projection_results_<subspace stem>_random.csv (grade + plot
                                                    #   locally the same way; identification of the paired concept is
                                                    #   then a false-positive control)
    modal run modal_app.py::run_projection --random-vectors --random-seed 1
                                                    # replicate with fresh random directions: non-zero seeds write to
                                                    #   projection_results_<...>_random_seed1.csv (seed 0 keeps the
                                                    #   plain _random name), so replicates never overwrite each other.
                                                    #   Download + grade each CSV separately, then pool at plot time:
                                                    #   python plot_projection.py --csv new_results/..._random.csv new_results/..._random_seed1.csv
    # Centered (conventional) PCA variant: every downstream default derives its name from the
    # .npz stem, so the _centered suffix propagates and never overwrites the uncentered files:
    modal run modal_app.py::run_pca --center        #   -> /results/pca_subspace_all_concepts_layer15_coeff6_centered.npz
    modal run modal_app.py::run_projection --subspace pca_subspace_all_concepts_layer15_coeff6_centered.npz
                                                    #   -> projection_results_pca_subspace_..._centered.csv
    # Judge locally and plot. The projection CSV + judge_question txt sit at the VOLUME ROOT
    # (run_projection runs with cwd=/results), and `modal volume get` needs that remote path
    # explicitly -- there is no bare `get <volume> .`:
    modal volume get introspection-results projection_results_pca_subspace_all_concepts_layer15_coeff6.csv ./new_results --force
    modal volume get introspection-results judge_question_projection_results_pca_subspace_all_concepts_layer15_coeff6.txt ./new_results --force
    python judge_results.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv --judges coherence affirmative_response affirmative_response_followed_by_correct_identification
    python plot_projection.py --csv new_results/projection_results_pca_subspace_all_concepts_layer15_coeff6.csv


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
def save_vectors(layers: str = "all", datasets: str = "simple_data", word_keys: str = ""):
    # "all" = omit --layer_range so save_vectors.py's default (layers 0-31) applies.
    # All layers cost the same as one: activations for every layer come from the
    # same single forward pass per prompt.
    layer_args = [] if layers == "all" else ["--layer_range", *layers.split()]
    # word_keys (simple_data only): which word lists of the json get vectors,
    # e.g. "concept_vector_words famous_people". Empty = concept_vector_words only.
    word_key_args = ["--word_keys", *word_keys.split()] if word_keys else []
    subprocess.run(
        [
            "python", "/root/app/save_vectors.py",
            *layer_args,
            "--datasets", *datasets.split(),
            *word_key_args,
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
    judges: str = "none",  # default: no OpenAI calls on the GPU; grade locally with judge_results.py
    run_name: str = "",
    max_batch_size: int = 25,
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
        "--max_batch_size", str(max_batch_size),
        "--skip_existing" if skip_existing else "--no_skip_existing",
    ]
    if concepts:
        # Comma-separated when any concept name contains spaces (e.g. "Albert Einstein");
        # plain space-separated still works for single-word concept lists.
        splitter = "," if "," in concepts else None
        cmd += ["--concepts", *[c.strip() for c in concepts.split(splitter)]]
    if judges:
        cmd += ["--judges", *judges.split()]
    if run_name:
        cmd += ["--run_name", run_name]
    # cwd=/results so main.py's relative new_results/ and plots/ land on the Volume
    subprocess.run(cmd, check=True, cwd="/results")
    results_vol.commit()


@app.function(
    image=image,
    timeout=1800,
    volumes=VOLUMES,  # reads the CSV from /results and vectors from /vectors; no GPU/secrets
)
def run_pca(
    csv: str = "new_results/output_all_concepts_layer15_coeff6.csv",
    layer: int = 15,
    vec_type: str = "avg",
    min_hits: int = 1,
    test_frac: float = 0.2,
    center: bool = False,
    out: str = "",
):
    # NOTE: grading happens locally, so the CSV on the Volume straight off the GPU has empty
    # judge columns and pca.py would find 0 identified concepts. Upload the GRADED copy first:
    #     modal volume put introspection-results new_results/output_...csv new_results/output_...csv --force
    # cwd=/root/app so pca.py's relative dataset/simple_data.json resolves; the CSV and the
    # .npz output live on the /results Volume, the vectors on /vectors.
    # Default output name mirrors pca.py's: derived from the CSV, with a _centered suffix
    # for center=True fits so they never overwrite the uncentered .npz on the Volume.
    if not out:
        from pathlib import Path
        run = Path(csv).stem.removeprefix("output_") + ("_centered" if center else "")
        out = f"pca_subspace_{run}.npz"
    cmd = [
        "python", "/root/app/pca.py",
        "--csv", f"/results/{csv}",
        "--vectors-dir", "/vectors/llama",
        "--layer", str(layer),
        "--vec-type", vec_type,
        "--min-hits", str(min_hits),
        "--test-frac", str(test_frac),
        "--out", f"/results/{out}",
    ]
    if center:
        cmd.append("--center")
    subprocess.run(cmd, check=True, cwd="/root/app")
    results_vol.commit()


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
def run_projection(
    subspace: str = "pca_subspace_all_concepts_layer15_coeff6.npz",
    ks: str = "1 2 5 10 20",
    interp_steps: int = 5,
    coeff: float = 6.0,
    temperature: float = 0.8,
    trials: int = 5,
    split: str = "test",
    judges: str = "none",  # default: generate only on the GPU; grade locally afterwards
    random_vectors: bool = False,  # random-direction control (see projection_experiment.py --random)
    random_seed: int = 0,
    out: str = "",
):
    # subspace .npz produced by run_pca lives on /results; vectors on /vectors. cwd=/results
    # so projection_experiment.py's default CSV output (and its judge_question_*.txt) land on
    # the Volume (imports still resolve via the script dir /root/app). judges='none' keeps the
    # GPU from idling on OpenAI calls; download + grade + plot locally (the CSV and its
    # judge_question_*.txt land at the volume root; get needs both explicitly):
    #     modal volume get introspection-results projection_results_<subspace stem>.csv . --force
    #     modal volume get introspection-results judge_question_projection_results_<subspace stem>.txt . --force
    #     python judge_results.py --csv projection_results_<subspace stem>.csv \
    #         --judges coherence affirmative_response affirmative_response_followed_by_correct_identification
    #     python plot_projection.py --csv projection_results_<subspace stem>.csv
    cmd = [
        "python", "/root/app/projection_experiment.py",
        "--subspace", f"/results/{subspace}",
        "--vectors-dir", "/vectors/llama",
        "--ks", *ks.split(),
        "--interp-steps", str(interp_steps),
        "--coeff", str(coeff),
        "--temperature", str(temperature),
        "--trials", str(trials),
        "--split", split,
    ]
    if random_vectors:
        # Random directions norm-matched to the held-out concepts: 'full' = base random
        # rate, proj_k / residual_k = random direction inside / orthogonal to the
        # subspace. Output gets a _random suffix (plus _seed<N> for non-zero seeds), so
        # neither the concept run nor earlier random replicates are ever overwritten.
        cmd += ["--random", "--random-seed", str(random_seed)]
    if judges:
        cmd += ["--judges", *judges.split()]
    if out:
        cmd += ["--out", f"/results/{out}"]
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

# Every word list of dataset/simple_data.json except the baseline pool: the 50 main
# concept_vector_words plus 5 categories x 50 words = 300 concepts total.
ALL_CONCEPT_WORD_KEYS = [
    "concept_vector_words", "famous_people", "countries",
    "concrete_nouns", "abstract_nouns", "verbs",
]


def load_all_concept_words():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).parent / "dataset" / "simple_data.json").read_text(encoding="utf-8"))
    return [word for key in ALL_CONCEPT_WORD_KEYS for word in data[key]]


@app.local_entrypoint()
def all_concepts(
    step: str = "all",
    temperature: float = 0.8,
    trials_per_cell: int = 5,
):
    """Coherence & affirmation over all 300 concepts at layer 15, alpha 6, vec_type avg.

    Concepts = every word list in dataset/simple_data.json except baseline_words: the
    50 main concept_vector_words + 5 categories x 50 (famous_people, countries,
    concrete_nouns, abstract_nouns, verbs). step='all' (default) first computes the
    missing layer-15 vectors for the 250 category words (already-saved vectors, i.e.
    the 50 main concepts, are skipped), then runs the experiment; 'vectors' /
    'experiment' runs just that half. Category names may contain spaces (e.g.
    "Albert Einstein"), so concepts are passed comma-separated. Generation only --
    judge afterwards with the two requested judges:

        python judge_results.py --csv new_results/output_all_concepts_layer15_coeff6.csv --judges coherence affirmative_response_followed_by_correct_identification

        modal run modal_app.py::all_concepts                                      # greedy, 1 trial/concept
        modal run modal_app.py::all_concepts --temperature 0.8 --trials-per-cell 5  # sampled distribution instead

    Results: new_results/output_all_concepts_layer15_coeff6.csv on the results Volume.
    """
    if step not in ("all", "vectors", "experiment"):
        raise ValueError(f"Unknown step: {step} (use 'all', 'vectors', or 'experiment')")
    concepts = load_all_concept_words()
    if step in ("all", "vectors"):
        print(f"Computing layer-15 vectors for word lists: {ALL_CONCEPT_WORD_KEYS}")
        save_vectors.remote(layers="15", word_keys=" ".join(ALL_CONCEPT_WORD_KEYS))
    if step in ("all", "experiment"):
        print(f"Running all-concepts experiment: {len(concepts)} concepts, layer=15, alpha=6, "
              f"vec_type=avg, temperature={temperature}, trials_per_cell={trials_per_cell}")
        run_experiment.remote(
            layers="15",
            coeffs="6",
            experiment_type="anthropic_reproduce",
            vec_types="avg",
            skip_existing=True,
            temperature=temperature,
            trials_per_cell=trials_per_cell,
            concepts=",".join(concepts),
            judges="none",
            run_name="all_concepts_layer15_coeff6",
        )


@app.local_entrypoint()
def control(
    samples: int = 100,
    temperature: float = 0.8,
    concept: str = "Dust",
    layer: str = "15",
):
    """Control: coherence & affirmation baseline with NO concept injected.

    Runs `samples` generations of the anthropic_reproduce prompt at coeff=0 (the
    steering vector is added with zero weight, i.e. no injection) and temperature
    `temperature`. This measures the baseline / false-positive rate when nothing
    is injected. Generation only -- judge afterwards with judge_results.py.

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
        skip_existing=True,
        temperature=temperature,
        trials_per_cell=samples,
        concepts=concept,  # coeff=0 => which concept is loaded doesn't affect output
        judges="none",
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
    The 5 trials of each cell run as one batched generate() call; generation only --
    judge afterwards with judge_results.py. skip_existing=True means a rerun resumes
    from whatever is already in the CSV on the Volume instead of starting over.

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
        skip_existing=True,
        temperature=temperature,
        trials_per_cell=trials_per_cell,
        concepts=concepts,
        judges="none",
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
